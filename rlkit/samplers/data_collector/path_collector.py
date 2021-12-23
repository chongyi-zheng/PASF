from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout


class ActionAgent():
    def __init__(self):

        self._actions = None
        self._step = 0

    def reset(self):
        self._step = 0

    def set_action(self, actions):
        self._actions = actions

    def get_action(self, *args, **kwargs):
        action = self._actions[self._step]
        self._step += 1
        return action, []


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self._action_agent = ActionAgent()

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def collect_aligned_paths(self, path_actions, discard_incomplete_paths=True):
        paths = []
        num_steps_collected = 0
        for p in path_actions:
            max_path_length = len(p)
            self._action_agent.set_action(p)
            path = self._rollout_fn(
                self._env,
                self._action_agent,
                max_path_length=max_path_length,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        # TODO (chongyi zheng): Do we need to save epoch path as well?
        snapshot_dict = dict(
            policy=self._policy,
            num_steps_total=self._num_steps_total,
            num_paths_total=self._num_paths_total,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict

    def load_from_snapshot(self, snapshot):
        # (chongyi zheng): Implement this for resuming
        self._policy = snapshot['policy']
        self._num_steps_total = snapshot['num_steps_total']
        self._num_paths_total = snapshot['num_paths_total']
        if self._save_env_in_snapshot:
            assert 'env' in snapshot
            if hasattr(self._env, '_custom_goal_sampler'):
                snapshot['env']._custom_goal_sampler = self._env._custom_goal_sampler
            self._env = snapshot['env']


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot

    def load_from_snapshot(self, snapshot):
        # (chongyi zheng): Implement this for resuming
        super().load_from_snapshot(snapshot)
        self._observation_key = snapshot['observation_key']
        self._desired_goal_key = snapshot['desired_goal_key']


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
