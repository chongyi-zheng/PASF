import abc
from collections import OrderedDict

import gtimer as gt
import os
import os.path as osp

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector

import torch
import random
import numpy as np

import subprocess


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_envs,
            evaluation_envs,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            aligned_buffer,
    ):
        self.trainer = trainer
        self.expl_envs = exploration_envs
        self.eval_envs = evaluation_envs
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self.aligned_buffer = aligned_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0, save_snapshot=False, save_intervals=10):
        self._start_epoch = start_epoch
        self._train(save_snapshot, save_intervals)

    def _train(self, save_snapshot=False, save_intervals=10):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch, save_snapshot=False, save_intervals=10):
        if save_snapshot and epoch % save_intervals == 0:
            snapshot = self._get_snapshot()
            replay_buffer_snapshot = dict()
            delete_keys = []
            for k, v in snapshot.items():
                if 'replay_buffer' in k:
                    for buffer_k, buffer_v in snapshot[k].items():
                        if 'observations' in buffer_k or 'next_observations' in buffer_k or \
                                'actions' in buffer_k or 'terminals' in buffer_k:
                            replay_buffer_snapshot[k.split('/')[-1] + '/' + buffer_k] = buffer_v
                            delete_keys.append((k, buffer_k))
            for k, buffer_k in delete_keys:
                del snapshot[k][buffer_k]
            aligned_buffer_snapshot = dict()
            delete_keys = []
            for k, v in snapshot.items():
                if 'aligned_buffer' in k:
                    for buffer_k, buffer_v in snapshot[k].items():
                        if 'observations' in buffer_k or 'next_observations' in buffer_k or \
                                'actions' in buffer_k or 'terminals' in buffer_k:
                            aligned_buffer_snapshot[k.split('/')[-1] + '/' + buffer_k] = buffer_v
                            delete_keys.append((k, buffer_k))
            for k, buffer_k in delete_keys:
                del snapshot[k][buffer_k]
            snapshot['torch_rng_state'] = torch.random.get_rng_state()
            snapshot['random_rng_state'] = random.getstate()
            snapshot['numpy_rng_state'] = np.random.get_state()

            ori_snapshot_dir = logger.get_snapshot_dir()
            ckpt_snapshot_dir = osp.join(ori_snapshot_dir, 'ckpt')
            os.makedirs(ckpt_snapshot_dir, exist_ok=True)
            logger.set_snapshot_dir(ckpt_snapshot_dir)
            logger.save_itr_params(epoch, snapshot)
            logger.save_extra_data(replay_buffer_snapshot,
                                   file_name='replay_buffer.npz', mode='numpy')
            logger.save_extra_data(aligned_buffer_snapshot,
                                   file_name='aligned_buffer.npz', mode='numpy')
            logger.save_extra_data(epoch, file_name='epoch.pkl', mode='pickle')
            process = subprocess.Popen(
                "cd {} && tar -cf ckpt.tmp.tar ckpt && rm -rf ckpt".format(
                    osp.abspath(ori_snapshot_dir)),
                shell=True, stdout=subprocess.PIPE)
            process.wait()
            logger.atomic_replace([osp.abspath(ckpt_snapshot_dir + '.tmp.tar')])
            logger.set_snapshot_dir(ori_snapshot_dir)
        gt.stamp('saving', unique=False)

        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.aligned_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {
            'expl_envs': self.expl_envs,
            'eval_envs': self.eval_envs,
        }
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        for k, v in self.aligned_buffer.get_snapshot().items():
            snapshot['aligned_buffer/' + k] = v
        return snapshot

    def _load_from_snapshot(self, snapshot):
        self.expl_envs = snapshot['expl_envs']
        self.eval_envs = snapshot['eval_envs']

        trainer_snapshot = dict()
        expl_data_collector_snapshot = dict()
        eval_data_collector_snapshot = dict()
        replay_buffer_snapshot = dict()
        aligned_buffer_snapshot = dict()

        for k, v in snapshot.items():
            if 'trainer' in k:
                trainer_snapshot[k.split('/')[-1]] = v
            elif 'exploration' in k:
                expl_data_collector_snapshot[k.split('/')[-1]] = v
            elif 'evaluation' in k:
                eval_data_collector_snapshot[k.split('/')[-1]] = v
            elif 'replay_buffer' in k:
                replay_buffer_snapshot[k.split('/')[-1]] = v
            elif 'aligned_buffer' in k:
                aligned_buffer_snapshot[k.split('/')[-1]] = v

        torch.random.set_rng_state(snapshot['torch_rng_state'])
        random.setstate(snapshot['random_rng_state'])
        np.random.set_state(snapshot['numpy_rng_state'])

        self.trainer.load_from_snapshot(trainer_snapshot)
        self.replay_buffer.load_from_snapshot(replay_buffer_snapshot)
        self.aligned_buffer.load_from_snapshot(aligned_buffer_snapshot)
        self.expl_data_collector.load_from_snapshot(expl_data_collector_snapshot)
        self.eval_data_collector.load_from_snapshot(eval_data_collector_snapshot)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        # save trainer_diagnostics to retrieve 'discount' later
        trainer_diagnostics = self.trainer.get_diagnostics()
        logger.record_dict(trainer_diagnostics, prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(epoch),
            prefix='exploration/'
        )

        for env in self.expl_envs:
            n = env.vae_env_name
            id = 'env-' + n[-2:] + '/'
            expl_paths = self.expl_data_collector.get_epoch_paths(n)
            if len(expl_paths) < 1:
                continue
            if hasattr(env, 'get_diagnostics'):
                logger.record_dict(
                    env.get_diagnostics(expl_paths,
                                        discount=trainer_diagnostics.get('discount', None)),
                    prefix='exploration/' + id,
                )
            # logger.record_dict(
            #     eval_util.get_generic_path_information(expl_paths),
            #     prefix="exploration/" + id,
            # )

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(epoch),
            prefix='evaluation/',
        )
        for env in self.eval_envs:
            n = env.vae_env_name
            id = 'env-' + n[-2:] + '/'
            eval_paths = self.eval_data_collector.get_epoch_paths(n)
            if hasattr(env, 'get_diagnostics'):
                logger.record_dict(
                    env.get_diagnostics(eval_paths,
                                        discount=trainer_diagnostics.get('discount', None)),
                    prefix='evaluation/'+id,
                )
            # logger.record_dict(
            #     eval_util.get_generic_path_information(eval_paths),
            #     prefix="evaluation/"+id,
            # )


        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False, write_header=(epoch == 0))

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
