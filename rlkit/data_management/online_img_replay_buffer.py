
from multiworld.core.image_env import normalize_image
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.shared_obs_dict_replay_buffer import \
    SharedObsDictRelabelingBuffer
from rlkit.data_management.obs_dict_replay_buffer import \
    normalize_image


class OnlineIMGBuffer(SharedObsDictRelabelingBuffer):

    def __init__(
            self,
            *args,
            env_name=None,
            **kwargs
    ):
        super().__init__(internal_keys=[], *args, **kwargs)
        self.buffer_env_name = env_name
        self.epoch = 0

    def add_path(self, path):
        super().add_path(path)

    def sample_indices(self, batch_size):
        return self._sample_indices(batch_size)

    def get_batch(self, idxs):
        next_obs = self._batch_next_obs_dict(idxs)
        next_obs = normalize_image(next_obs[self.observation_key])
        return dict(
            next_obs=ptu.from_numpy(next_obs),
        )

    def get_diagnostics(self):
        return {}



