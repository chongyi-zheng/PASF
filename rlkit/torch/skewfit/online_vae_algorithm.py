import gtimer as gt
from rlkit.core import logger
from rlkit.data_management.GBVAEBuffer import \
    GoalBlockVAEBuffer
from rlkit.data_management.shared_obs_dict_replay_buffer \
    import SharedObsDictRelabelingBuffer
import rlkit.torch.vae.vae_schedules as vae_schedules
import rlkit.torch.vae.vae_sampler_schedule as vae_sampler
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
import rlkit.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread


class OnlineVaeAlgorithm(TorchBatchRLAlgorithm):

    def __init__(
            self,
            vae,
            vae_trainer,
            *base_args,
            vae_save_period=1,
            vae_training_schedule=vae_schedules.never_train,
            vae_sampler_schedule=vae_sampler.random_only,
            oracle_data=False,
            parallel_vae_train=True,
            vae_min_num_steps_before_training=0,
            uniform_dataset=None,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        assert isinstance(self.replay_buffer, GoalBlockVAEBuffer) and \
               isinstance(self.aligned_buffer, GoalBlockVAEBuffer)

        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.vae_sampler_schedule = vae_sampler_schedule
        self.oracle_data = oracle_data

        self.parallel_vae_train = parallel_vae_train
        self.vae_min_num_steps_before_training = vae_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

        self._vae_training_process = None
        self._update_subprocess_vae_thread = None
        self._vae_conn_pipe = None

    def _train(self, save_snapshot=False, save_intervals=10):
        # set up for training
        self.vae.eval()
        self.training_mode(False)
        super()._train(save_snapshot, save_intervals)
        self._cleanup()

    def _end_epoch(self, epoch, save_snapshot=False, save_intervals=10):
        self._train_vae(epoch)
        gt.stamp('vae training')
        super()._end_epoch(epoch, save_snapshot, save_intervals)

    def _log_stats(self, epoch):
        self._log_vae_stats()
        super()._log_stats(epoch)

    def to(self, device):
        self.vae.to(device)
        super().to(device)

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        assert 'vae' not in snapshot
        snapshot['vae'] = self.vae
        # (chongyi zheng): save vae trainer for resuming, note that vae_trainer is not directly pickleable
        snapshot['vae_trainer'] = self.vae_trainer.get_snapshot()
        return snapshot

    """
    VAE-specific Code
    """
    def _train_vae(self, epoch):
        if self.parallel_vae_train and self._vae_training_process is None:
            self.init_vae_training_subprocess()
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        rl_start_epoch = 5
        if should_train or epoch <= (rl_start_epoch - 1):
            if self.parallel_vae_train:
                assert self._vae_training_process.is_alive()
                # Make sure the last vae update has finished before starting
                # another one
                if self._update_subprocess_vae_thread is not None:
                    self._update_subprocess_vae_thread.join()
                self._update_subprocess_vae_thread = Thread(
                    target=OnlineVaeAlgorithm.update_vae_in_training_subprocess,
                    args=(self, epoch, ptu.device)
                )
                self._update_subprocess_vae_thread.start()
                self._vae_conn_pipe.send((amount_to_train, epoch))
            else:
                random_sampler, aligned_sampler = self.sampler_schedule(epoch)

                _train_vae(
                    self.vae_trainer,
                    random_sampler=random_sampler,
                    aligned_sampler=aligned_sampler,
                    epoch=epoch,
                    batches=amount_to_train
                )
                self.refresh_latents(epoch)
                _test_vae(
                    self.vae_trainer,
                    epoch,
                    self.replay_buffer,
                    vae_save_period=self.vae_save_period,
                    uniform_dataset=self.uniform_dataset,
                )


    def refresh_latents(self, epoch):
        self.replay_buffer.refresh_latents(epoch)
        if self.use_aligned_buffer:
            self.aligned_buffer.refresh_latents(epoch)

    def sampler_schedule(self, epoch):
        if not self.use_aligned_buffer:
            return self.replay_buffer.random_vae_training_data, None
        else:
            random, align = self.vae_sampler_schedule(epoch)
            random = self.sampler_post_processing(random)
            align = self.sampler_post_processing(align)
            return random, align

    def sampler_post_processing(self, sampler_id):
        if sampler_id is 'none':
            return None
        elif sampler_id is 'random':
            return self.replay_buffer.random_vae_training_data
        elif sampler_id is 'align':
            return self.aligned_buffer.random_aligned_training_data

    def _log_vae_stats(self):
        logger.record_dict(
            self.vae_trainer.get_diagnostics(),
            prefix='vae_trainer/',
        )

    def _cleanup(self):
        if self.parallel_vae_train:
            self._vae_conn_pipe.close()
            self._vae_training_process.terminate()

    def init_vae_training_subprocess(self):
        assert isinstance(self.replay_buffer, SharedObsDictRelabelingBuffer)

        self._vae_conn_pipe, process_pipe = Pipe()
        self._vae_training_process = Process(
            target=subprocess_train_vae_loop,
            args=(
                process_pipe,
                self.vae,
                self.vae.state_dict(),
                self.replay_buffer,
                self.replay_buffer.get_mp_info(),
                ptu.device,
            )
        )
        self._vae_training_process.start()
        self._vae_conn_pipe.send(self.vae_trainer)

    def update_vae_in_training_subprocess(self, epoch, device):
        self.vae.__setstate__(self._vae_conn_pipe.recv())
        self.vae.to(device)
        _test_vae(
            self.vae_trainer,
            epoch,
            self.replay_buffer,
            vae_save_period=self.vae_save_period,
            uniform_dataset=self.uniform_dataset,
        )

    def load(self, epoch):
        snapshot = logger.load_itr_params(epoch)
        snapshot['torch_rng_state'] = snapshot['torch_rng_state'].cpu()

        replay_buffer_snapshot = logger.load_extra_data('replay_buffer.npz', mode='numpy')
        for k, v in replay_buffer_snapshot.items():
            buffer_k = k.replace(k.split('/')[0] + '/', '')
            snapshot['replay_buffer/' + k.split('/')[0]][buffer_k] = v
        # (chongyi zheng): load aligned buffer
        aligned_buffer_snapshot = logger.load_extra_data('aligned_buffer.npz', mode='numpy')
        for k, v in aligned_buffer_snapshot.items():
            buffer_k = k.replace(k.split('/')[0] + '/', '')
            snapshot['aligned_buffer/' + k.split('/')[0]][buffer_k] = v
        super()._load_from_snapshot(snapshot)
        assert 'vae' in snapshot
        assert 'vae_trainer' in snapshot
        self.vae = snapshot['vae']
        self.vae_trainer.load_from_snapshot(snapshot['vae_trainer'])


def _train_vae(vae_trainer, random_sampler, aligned_sampler, epoch, batches=50, oracle_data=False):
    if oracle_data:
        batch_sampler = None
    vae_trainer.train_epoch(
        epoch,
        random_batch=random_sampler,
        aligned_batch=aligned_sampler,
        batches=batches,
        from_rl=True,
    )


def _test_vae(vae_trainer, epoch, replay_buffer, vae_save_period=1, uniform_dataset=None):
    save_imgs = epoch % vae_save_period == 0
    log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    if uniform_dataset is not None:
        replay_buffer.log_loss_under_uniform(uniform_dataset, vae_trainer.batch_size, rl_logger=vae_trainer.vae_logger_stats_for_rl)
    vae_trainer.test_epoch(
        epoch,
        from_rl=True,
        save_reconstruction=False,
    )
    if save_imgs:
        vae_trainer.dump_prior_samples(epoch)
        vae_trainer.dump_oracle_samples(epoch)
        if log_fit_skew_stats:
            replay_buffer.dump_best_reconstruction(epoch)
            replay_buffer.dump_worst_reconstruction(epoch)
            replay_buffer.dump_sampling_histogram(epoch, batch_size=vae_trainer.batch_size)
        if uniform_dataset is not None:
            replay_buffer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)


def subprocess_train_vae_loop(
        conn_pipe,
        vae,
        vae_params,
        replay_buffer,
        mp_info,
        device,
):
    """
    The observations and next_observations of the replay buffer are stored in
    shared memory. This loop waits until the parent signals to start vae
    training, trains and sends the vae back, and then refreshes the latents.
    Refreshing latents in the subprocess reflects in the main process as well
    since the latents are in shared memory. Since this is does asynchronously,
    it is possible for the main process to see half the latents updated and half
    not.
    """
    ptu.device = device
    vae_trainer = conn_pipe.recv()
    vae.load_state_dict(vae_params)
    vae.to(device)
    vae_trainer.set_vae(vae)
    replay_buffer.init_from_mp_info(mp_info)
    replay_buffer.env.vae = vae
    while True:
        amount_to_train, epoch = conn_pipe.recv()
        _train_vae(vae_trainer, replay_buffer, epoch, amount_to_train)
        conn_pipe.send(vae_trainer.model.__getstate__())
        replay_buffer.refresh_latents(epoch)
