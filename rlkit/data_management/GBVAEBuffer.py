
from rlkit.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
import copy
import numpy as np
import torch
from collections import OrderedDict


class GoalBlockVAEBuffer:
    def __init__(self, vae, envs, observation_key, desired_goal_key, achieved_goal_key, variant):
        self.n_envs = len(envs)
        self.env_names = [n.vae_env_name for n in envs]
        self.env_idx = {}

        buffers = dict()
        for env in envs:
            ot = np.eye(self.n_envs)[int(env.vae_env_name[-2:])].reshape(1, -1)
            self.env_idx[env.vae_env_name] = ot
            replay_buffer = OnlineVaeRelabelingBuffer(vae=vae, env=env, env_idx=ot,
                                                      env_name=env.vae_env_name,
                                                      observation_key=observation_key,
                                                      desired_goal_key=desired_goal_key,
                                                      achieved_goal_key=achieved_goal_key,
                                                      **variant)

            buffers[env.vae_env_name] = replay_buffer

        self._prioritize_vae_samples = buffers[self.env_names[0]]._prioritize_vae_samples
        self.buffers = buffers

    def add_paths(self, paths):
        for n, path in paths.items():
            self.buffers[n].add_paths(path)

    def random_batch(self, batch_size):
        bs_i = int(batch_size / self.n_envs)
        batches = list()
        for b in self.buffers.values():
            batches.append(b.random_batch(bs_i))

        batch = copy.deepcopy(batches[0])
        if self.n_envs > 1:
            for b in batches[1:]:
                for k in b.keys():
                    batch[k] = np.concatenate((batch[k], b[k]), axis=0)
        return batch

    def random_vae_training_data(self, batch_size):
        bs_i = int(batch_size / self.n_envs)
        batch = {}
        for n, buffer in self.buffers.items():
            batch[n] = buffer.random_vae_training_data(bs_i)

        return batch

    def random_aligned_training_data(self, batch_size):
        bs = int(batch_size / self.n_envs)
        r = int(np.random.randint(self.n_envs, size=(1,)))
        env_id = self.env_names[r]
        idxs = self.buffers[env_id].sample_weighted_indices(bs)
        batch = {}
        for n, buffer in self.buffers.items():
            batch[n] = buffer.get_batch(idxs)
        return batch

    def refresh_latents(self, epoch):
        for buffer in self.buffers.values():
            buffer.refresh_latents(epoch)

    def get_snapshot(self):  # get model snapshot
        snapshot_dict = dict()
        for name, buffer in self.buffers.items():
            snapshot = buffer.get_snapshot()
            snapshot_dict[name] = snapshot

        return snapshot_dict

    def get_diagnostics(self):  # get diagnostic information for logging
        od = OrderedDict()
        for name, buffer in self.buffers.items():
            diagnostic = buffer.get_diagnostics()
            for k, v in diagnostic.items():
                od['env-' + name[-1] + '/' + k] = v
        return od

    def end_epoch(self, epoch):
        return

    def load_from_snapshot(self, snapshot):
        for name, buffer in self.buffers.items():
            assert hasattr(buffer, 'load_from_snapshot')
            buffer.load_from_snapshot(snapshot[name])
