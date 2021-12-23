
from rlkit.data_management.online_img_replay_buffer import OnlineIMGBuffer
from collections import OrderedDict

class GoalBlockIMGBuffer():
    def __init__(self, envs, variant,
                 observation_key='image_observation',
                 desired_goal_key='image_desired_goal',
                 achieved_goal_key='image_achieved_goal'
                 ):

        self.envs = envs
        self.n_envs = len(envs)
        self.env_names = [n.vae_env_name for n in envs]

        buffers = dict()
        for env in envs:
            replay_buffer = OnlineIMGBuffer(env=env, env_name=env.vae_env_name,
                                                observation_key=observation_key,
                                                desired_goal_key=desired_goal_key,
                                                achieved_goal_key=achieved_goal_key,
                                                **variant)
            buffers[env.vae_env_name] = replay_buffer

        self.buffers = buffers

    def add_paths(self, paths):
        for n, path in paths.items():
            self.buffers[n].add_paths(path)

    def random_vae_training_data(self, batch_size, epoch=None):
        bs = int(batch_size / self.n_envs)
        idxs = self.sample_indices(bs)
        batch = {}
        for n, buffer in self.buffers.items():
            batch[n] = buffer.get_batch(idxs)
        return batch

    def sample_indices(self, batch_size):
        return self.buffers[self.env_names[0]].sample_indices(batch_size)

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        od = OrderedDict()
        for n, buffer in self.buffers.items():
            diagnostic = buffer.get_diagnostics()
            for k, v in diagnostic.items():
                od['env-' + n[-2:] + '/' + k] = v
        return od

    def end_epoch(self, epoch):
        return


