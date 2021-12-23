from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
import cv2

from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset,
    InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)

from rlkit.util.ml_util import ConstantSchedule
from rlkit.torch.vae.mmd import MMDEstimator


def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    log_probs = log_probs - log_probs.mean()
    log_probs = np.clip(log_probs, -40., 40.)
    probs = np.exp(log_probs)
    assert not np.any(probs <= 0), 'choose a smaller power'
    return probs

def compute_log_p_log_q_log_d(
    model,
    data,
    idxs,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    batch_size = data.shape[0]
    idxs = torch.repeat_interleave(idxs, batch_size, dim=0)
    representation_size = model.representation_size
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, representation_size)),
                        ptu.ones((batch_size, representation_size)))
    mus, logvars = latent_distribution_params
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)
        if decoder_distribution == 'bernoulli':
            latents = torch.cat([latents, idxs], dim=-1)
            decoded = model.decode(latents)[0]
            log_d_x_given_z = torch.log(imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=1)
        elif decoder_distribution == 'gaussian_identity_variance':
            latents = torch.cat([latents, idxs], dim=-1)
            _, obs_distribution_params = model.decode(latents)
            dec_mu, dec_logvar = obs_distribution_params
            dec_var = dec_logvar.exp()
            decoder_dist = Normal(dec_mu, dec_var.pow(.5))
            log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=1)
        else:
            raise EnvironmentError('Invalid Decoder Distribution Provided')

        log_p[:, i] = log_p_z
        log_q[:, i] = log_q_z_given_x
        log_d[:, i] = log_d_x_given_z
    return log_p, log_q, log_d

def compute_p_x_np_to_np(
    model,
    data,
    power,
    idxs,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    assert power >= -1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
        ptu.from_numpy(idxs),
        decoder_distribution,
        num_latents_to_sample,
        sampling_method
    )

    if sampling_method == 'importance_sampling':
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
    elif sampling_method == 'biased_sampling' or sampling_method == 'true_prior_sampling':
        log_p_x = log_d.mean(dim=1)
    else:
        raise EnvironmentError('Invalid Sampling Method Provided')
    log_p_x_skewed = power * log_p_x
    return ptu.get_numpy(log_p_x_skewed)



class FairVAETrainer(object):
    def __init__(
            self,
            img_dataset,
            idx_dataset,
            oracle_dataset,
            model,
            envs,
            random_batch_size=96,
            aligned_batch_size=96,
            log_interval=0,
            beta=0.0,
            c_mmd=0.0,
            c_diff=0.0,
            D=1024,
            negative_samples=5,
            beta_schedule=None,
            c_mmd_schedule=None,
            c_diff_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            train_data_workers=0,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
    ):
        if skew_config is None:
            skew_config = {}
        self.log_interval = log_interval
        self.random_batch_size = random_batch_size
        self.aligned_batch_size = aligned_batch_size
        self.beta = beta
        self.c_mmd = c_mmd
        self.c_diff = c_diff
        if is_auto_encoder:
            self.beta = 0
        if lr is None:
            if is_auto_encoder:
                lr = 1e-2
            else:
                lr = 1e-3
        self.beta_schedule = beta_schedule
        self.c_mmd_schedule = c_mmd_schedule
        self.c_diff_schedule = c_diff_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        if self.c_mmd_schedule is None:
            self.c_mmd_schedule = ConstantSchedule(self.c_mmd)
        if self.c_diff_schedule is None:
            self.c_diff_schedule = ConstantSchedule(self.c_diff)

        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot
        self.env_idxs = {}
        self.envs = envs
        self.env_names = [env.vae_env_name for env in envs]
        self.n_envs = len(envs)

        for env in envs:
            n = env.vae_env_name
            ot = np.eye(self.n_envs)[int(n[-2:])].reshape(1, -1)
            self.env_idxs[n] = ptu.from_numpy(ot)

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        self.params = list(self.model.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.lr,
            weight_decay=weight_decay)

        self.MMD = MMDEstimator(self.representation_size, D)
        self.negative_samples = negative_samples
        self.img_dataset, self.idx_dataset = img_dataset, idx_dataset
        self.oracle_dataset = oracle_dataset
        assert self.img_dataset.dtype == np.uint8

        self.train_data_workers = train_data_workers
        self.skew_dataset = skew_dataset
        self.skew_config = skew_config
        self.start_skew_epoch = start_skew_epoch
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        if self.skew_dataset:
            self._train_weights = self._compute_train_weights()
        else:
            self._train_weights = None

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        self.eval_statistics = OrderedDict()
        self._extra_stats_to_log = None

    def get_dataset_stats(self, data):
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def _compute_train_weights(self):
        method = self.skew_config.get('method', 'squared_error')
        power = self.skew_config.get('power', 1)
        size = self.img_dataset.shape[0]
        weights = np.zeros(size)

        idxs = np.arange(size)
        img_data = self.img_dataset[idxs]
        if method == 'vae_prob':
            img_data = normalize_image(img_data)
            weights[idxs] = compute_p_x_np_to_np(self.model, img_data, power=power, **self.priority_function_kwargs)
        else:
            raise NotImplementedError('Method {} not supported'.format(method))

        if method == 'vae_prob':
            weights = relative_probs_from_log_probs(weights)
        return weights

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_batch(self, **kwargs):
        ids = np.random.randint(self.img_dataset.shape[0], size=10)
        imgs, env_idxs = self.img_dataset[ids], self.idx_dataset[ids]
        return ptu.from_numpy(normalize_image(imgs)), ptu.from_numpy(env_idxs)

    def train_epoch(self, epoch, random_batch, aligned_batch=None, batches=100, from_rl=False):
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        mmds = []
        diffs = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        c_mmd = float(self.c_mmd_schedule.get_value(epoch))
        c_diff = float(self.c_diff_schedule.get_value(epoch))

        for batch_idx in range(batches):
            log_prob = 0.
            kle = 0.
            latent_means = []
            latent_idxs = []
            aligned_latents = []
            random_data = random_batch(self.random_batch_size)
            aligned_data = aligned_batch(self.aligned_batch_size) \
                if aligned_batch is not None else None

            for n, b in random_data.items():
                next_obs = b['next_obs']
                idx = torch.repeat_interleave(self.env_idxs[n], next_obs.shape[0], dim=0)
                reconstruction_i, obs_distribution_param_i, latent_distribution_param_i = self.model([next_obs, idx])
                log_prob_i = self.model.logprob(next_obs, obs_distribution_param_i)
                kle_i = self.model.kl_divergence(latent_distribution_param_i)
                kle += kle_i
                log_prob += log_prob_i
                latent_mu = latent_distribution_param_i[0]
                latent_means.append(latent_mu)
                latent_idxs.append(idx)

            kle, log_prob = kle / len(random_data), log_prob / len(random_data)

            if aligned_data is not None:
                for n, b in aligned_data.items():
                    next_obs = b['next_obs']
                    latents = self.model.encode(next_obs)[0]
                    aligned_latents.append(latents)
                mmd_loss = self.MMD.forward(aligned_latents)
                difference_loss = self.difference_loss(aligned_latents)
            else:
                mmd_loss = self.MMD.forward(latent_means)
                difference_loss = self.difference_loss(latent_means)

            encoder_mean = torch.cat(latent_means, dim=0)

            z_data = ptu.get_numpy(encoder_mean.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])

            loss = -1 * log_prob + beta * kle + c_mmd * mmd_loss - c_diff * difference_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            kles.append(kle.item())
            log_probs.append(log_prob.item())
            mmds.append(mmd_loss.item())
            diffs.append(difference_loss.item())

        if not from_rl:
            zs = np.array(zs)
            self.model.dist_mu = zs.mean(axis=0)
            self.model.dist_std = zs.std(axis=0)

        self.eval_statistics['train/log prob'] = np.mean(log_probs)
        self.eval_statistics['train/KL'] = np.mean(kles)
        self.eval_statistics['train/MMD norm'] = np.mean(mmds)
        self.eval_statistics['train/loss'] = np.mean(losses)
        self.eval_statistics['train/Diff loss'] = np.mean(diffs)

        self.model.eval()

    def difference_loss(self, latents):
        total_loss = 0.
        for latent in latents:
            latent_2 = []
            for _ in range(self.negative_samples):  # (chongyi zheng): negative_samples * aligned_batch_size = diff_sample_size
                perm = np.random.permutation(len(latent))
                latent_2.append(latent[perm])
            latent_2 = torch.cat(latent_2, dim=0)
            latent = torch.cat([latent for _ in range(self.negative_samples)], dim=0)
            difference = torch.norm(latent - latent_2, p=2, dim=1)
            total_loss += difference.mean()

        return total_loss / len(latents)

    def get_diagnostics(self):
        return self.eval_statistics

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
    ):
        self.model.eval()
        losses = []
        log_probs = []
        kles = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(1):
            next_obs, idxs = self.get_batch(train=False)
            reconstructions, obs_distribution_params, latent_distribution_params = self.model([next_obs, idxs])
            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            loss = -1 * log_prob + beta * kle

            encoder_mean = latent_distribution_params[0]
            z_data = ptu.get_numpy(encoder_mean.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.size(0), 12)
                comparison = torch.cat([
                    next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
                        .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ).transpose(2, 3),
                    reconstructions.view(-1, self.input_channels,
                        self.imsize, self.imsize)[:n].transpose(2, 3),
                ])
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)


        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['test/log prob'] = np.mean(log_probs)
        self.eval_statistics['test/KL'] = np.mean(kles)
        self.eval_statistics['test/loss'] = np.mean(losses)
        self.eval_statistics['beta'] = beta
        if not from_rl:
            for k, v in self.eval_statistics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)

    def debug_statistics(self):

        debug_batch_size = 64
        data = self.get_batch(train=False)
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        if self.skew_dataset:
            stats.update(create_stats_ordered_dict(
                'train weight',
                self._train_weights
            ))
        return stats

    def get_snapshot(self):
        # TODO (chongyi zheng): We need to reconstruct 'train_dataloader' and 'test_dataloader' manually
        snapshot = dict(
            env_idxs=self.env_idxs,
            envs=self.envs,
            model=self.model,
            params=self.params,
            optimizer=self.optimizer,
            MMD=self.MMD,
            img_dataset=self.img_dataset,
            idx_dataset=self.idx_dataset,
            oracle_dataset=self.oracle_dataset,
        )

        return snapshot

    def load_from_snapshot(self, snapshot):
        # (chongyi zheng): Implement this for resuming
        self.env_idxs = snapshot['env_idxs']
        self.envs = snapshot['envs']
        self.model = snapshot['model']
        self.params = snapshot['params']
        self.optimizer = snapshot['optimizer']
        self.MMD = snapshot['MMD']
        self.img_dataset = snapshot['img_dataset']
        self.idx_dataset = snapshot['idx_dataset']
        self.oracle_dataset = snapshot['oracle_dataset']

    def dump_prior_samples(self, epoch):
        latent_sample = ptu.randn(64, self.representation_size)
        idx_sample = ptu.from_numpy(np.eye(self.n_envs)[np.random.
                                    randint(self.n_envs, size=(64, ))])
        shuffle_idx_sample = torch.cat([idx_sample[:, 1:], idx_sample[:, :1]], dim=-1)
        sample = torch.cat([latent_sample, idx_sample], dim=-1)
        shuffle_sample = torch.cat([latent_sample, shuffle_idx_sample], dim=-1)
        sample = self.model.decode(sample)[0].cpu()
        shuffle_sample = self.model.decode(shuffle_sample)[0].cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        shuffle_save_dir = osp.join(logger.get_snapshot_dir(), 'ss%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )
        save_image(
            shuffle_sample.data.view(64, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            shuffle_save_dir
        )

    def dump_oracle_samples(self, epoch):
        for n, data in self.oracle_dataset.items():
            if n not in self.env_names:
                continue
            max_len = min(100, len(data))
            imgs = ptu.from_numpy(data[:max_len])
            idxs = torch.repeat_interleave(self.env_idxs[n], max_len, dim=0)
            shuffled_idxs = torch.cat([idxs[:, 1:], idxs[:, :1]], dim=-1)
            reconstructions, _, _= self.model([imgs, idxs])
            shuffled_reconstructions, _, _ = self.model([imgs, shuffled_idxs])
            comparison = torch.cat([
                imgs.narrow(start=0, length=self.imlength, dim=1)
                    .contiguous().view(
                    -1, self.input_channels, self.imsize, self.imsize
                ).transpose(2, 3),
                reconstructions.view(-1, self.input_channels,
                                     self.imsize, self.imsize).transpose(2, 3),
                shuffled_reconstructions.view(-1, self.input_channels,
                                     self.imsize, self.imsize).transpose(2, 3)
            ])
            save_dir = osp.join(logger.get_snapshot_dir(),
                                n+'-o%d.png' % epoch)
            save_image(comparison.data.cpu(), save_dir, nrow=max_len)


    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.img_dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=len(idxs),
        )

    def log_loss_under_uniform(self, model, data, priority_function_kwargs):
        import torch.nn.functional as F
        log_probs_prior = []
        log_probs_biased = []
        log_probs_importance = []
        kles = []
        mses = []
        for i in range(0, data.shape[0], self.random_batch_size):
            img = normalize_image(data[i:min(data.shape[0], i + self.random_batch_size), :])
            torch_img = ptu.from_numpy(img)
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(torch_img)

            priority_function_kwargs['sampling_method'] = 'true_prior_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_prior = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'biased_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_biased = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'importance_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_importance = (log_p - log_q + log_d).mean()

            kle = model.kl_divergence(latent_distribution_params)
            mse = F.mse_loss(torch_img, reconstructions, reduction='elementwise_mean')
            mses.append(mse.item())
            kles.append(kle.item())
            log_probs_prior.append(log_prob_prior.item())
            log_probs_biased.append(log_prob_biased.item())
            log_probs_importance.append(log_prob_importance.item())

        logger.record_tabular("Uniform Data Log Prob (True Prior)", np.mean(log_probs_prior))
        logger.record_tabular("Uniform Data Log Prob (Biased)", np.mean(log_probs_biased))
        logger.record_tabular("Uniform Data Log Prob (Importance)", np.mean(log_probs_importance))
        logger.record_tabular("Uniform Data KL", np.mean(kles))
        logger.record_tabular("Uniform Data MSE", np.mean(mses))

    def dump_uniform_imgs_and_reconstructions(self, dataset, epoch):
        idxs = np.random.choice(range(dataset.shape[0]), 4)
        filename = 'uniform{}.png'.format(epoch)
        imgs = []
        recons = []
        for i in idxs:
            img_np = dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )

def img_show(img, imsize):
    img = img.reshape(3, imsize, imsize).transpose()
    img = img[::-1, :, ::-1]
    cv2.imshow('img', img)
    cv2.waitKey(5000)