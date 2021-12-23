import numpy as np
from os import path as osp
from rlkit.util.io import load_local_or_remote_file
from rlkit.core import logger
import cv2

from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.samplers.data_collector import GoalConditionedPathCollector

class GBVAEEnvPathCollector():
    def __init__(
            self,
            goal_sampling_mode,
            envs,
            policy,
            tsne_kwargs=None,
            oracle_dataset={},
            decode_goals=False,
            observation_key=None,
            desired_goal_key=None,
    ):
        self.vae = envs[0].vae
        self.env_names = [n.vae_env_name for n in envs]
        self.n_envs = len(envs)
        path_collectors = dict()
        for env in envs:
            pc = VAEWrappedEnvPathCollector(
                goal_sampling_mode=goal_sampling_mode,
                env=env,
                policy=policy,
                decode_goals=decode_goals,
                observation_key=observation_key,
                desired_goal_key=desired_goal_key)
            path_collectors[env.vae_env_name] = pc
        self.path_collectors = path_collectors


        self.use_tsne = False
        self.oracle_dataset = oracle_dataset
        self.tsne_save_period = 10

        if tsne_kwargs is not None:
            self.use_tsne = tsne_kwargs['use_tsne']
            self.tsne_save_period = tsne_kwargs['save_period']

    def get_oracle_latent_data(self):
        latent_data = {}
        for n, dataset in self.oracle_dataset.items():
            latent = self.vae.encode(ptu.from_numpy(dataset))[0]
            latent_data[n] = ptu.get_numpy(latent)
        return latent_data

    def get_latent_stats(self):
        stats = {}
        latent_data = self.get_oracle_latent_data()

        if len(latent_data) < 1:
            return {}, {}

        reference_latents = latent_data[self.env_names[0]]
        for n, latents in latent_data.items():
            error_norm = np.linalg.norm(latents - reference_latents, ord=2, axis=1)
            latent_norm = np.linalg.norm(latents, ord=2, axis=1)
            stats[n] = {'err': error_norm, 'latent': latent_norm}
        return stats, latent_data

    def collect_new_paths_subset(self, *args, env_names, **kwargs):
        paths = dict()
        for n, pc in self.path_collectors.items():
            if n in env_names:
                path = pc.collect_new_paths(*args, **kwargs)
                paths[n] = path
        return paths

    def end_epoch_subset(self, env_names, epoch):
        for n, pc in self.path_collectors.items():
            if n in env_names:
                pc.end_epoch(epoch)

    def collect_new_paths(self, *args, **kwargs):
        return self.collect_new_paths_subset(*args, env_names=self.env_names, **kwargs)

    def end_epoch(self, epoch):
        self.end_epoch_subset(env_names=self.env_names, epoch=epoch)

    def collect_aligned_paths(self, actions, paths):
        for n, pc in self.path_collectors.items():
            if n in paths:
                continue
            path = pc.collect_aligned_paths(actions)
            paths[n] = path
        return paths

    def collect_random_aligned_paths(self, *args, **kwargs):
        actions, paths = [], {}
        r = int(np.random.randint(self.n_envs, size=(1,)))
        env_id = self.env_names[r]
        path = self.path_collectors[env_id].collect_new_paths(*args, **kwargs)
        paths[env_id] = path
        for p in path:
            actions.append(p['actions'])

        return self.collect_aligned_paths(actions, paths)

    def get_snapshot(self):
        # TODO (chongyi zheng): use exact env name for easy resuming
        snapshot = dict(
            vae=self.vae,
            oracle_dataset=self.oracle_dataset
        )
        for n, pc in self.path_collectors.items():
            # id = 'env-' + name[-1] + '/'
            # for k, v in pc.get_snapshot().items():
            #     snapshot[id + k] = v
            snapshot[n] = pc.get_snapshot()
        return snapshot

    def get_diagnostics(self, epoch=0):
        od = dict()
        num_paths, num_steps = 0, 0
        for n, pc in self.path_collectors.items():
            id = 'env-' + n[-2:] + '/'
            for k, v in pc.get_diagnostics().items():
                od[id + k] = v
                if k == 'num paths total':
                    num_paths += v
                if k == 'num steps total':
                    num_steps += v
        od['all num paths total'] = num_paths
        od['all num steps total'] = num_steps

        stats, latent_data = self.get_latent_stats()
        for n, stat in stats.items():
            od[n[-2:] + '/latent err'] = stat['err'].mean()
            od[n[-2:] + '/latent norm'] = stat['latent'].mean()
            od[n[-2:] + '/latent err rate'] = \
                (stat['err'] / np.maximum(stat['latent'], 1E-2)).mean()
        self.dump_tsne_data(latent_data, epoch)

        return od

    def get_epoch_paths(self, n):
        return self.path_collectors[n].get_epoch_paths()

    def dump_tsne_data(self, data, epoch):
        if self.use_tsne and epoch % self.tsne_save_period == 0:
            np.save(osp.join(logger.get_snapshot_dir(), 'latent%d' % epoch), data)

    def load_from_snapshot(self, snapshot):
        # (chongyi zheng): Implement this for resuming
        self.oracle_dataset = snapshot['oracle_dataset']
        self.vae = snapshot['vae']
        for n, pc in self.path_collectors.items():
            pc.load_from_snapshot(snapshot[n])


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            goal_sampling_mode,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._goal_sampling_mode = goal_sampling_mode
        self._decode_goals = decode_goals
        self.pc_env_name = env.vae_env_name

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)

    def collect_aligned_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_aligned_paths(*args, **kwargs)

