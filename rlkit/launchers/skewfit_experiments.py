import os.path as osp
from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.envs.vae_wrapper import temporary_mode

import matplotlib.pyplot as plt
import cv2
import numpy as np

import subprocess

from rlkit.samplers.data_collector.vae_env import (
    GBVAEEnvPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.video import dump_video
from rlkit.util.ml_util import LinearSchedule


def skewfit_full_experiment(variant):
    variant['skewfit_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    # (chongyi zheng): Some potential bugs with initial VAE training may exist
    train_vae_and_update_variant(variant)
    skewfit_experiment(variant['skewfit_variant'])


def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    skewfit_variant = variant['skewfit_variant']
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    expl_env_num = variant.get('expl_env_num', 3)
    # (chongyi zheng): read checkpoint variants
    checkpoint = variant.get('checkpoint', False)
    checkpoint_dir = variant.get('checkpoint_dir', None)
    save_snapshot = variant.get('save_snapshot', False)
    save_intervals = variant.get('save_intervals', 10)

    if 'env_ids' in variant:
        assert 'env_class' not in variant
        skewfit_variant['env_ids'] = variant['env_ids']
        train_vae_variant['generate_vae_dataset_kwargs']['env_ids'] = \
            variant['env_ids'][:expl_env_num]
    else:
        raise ValueError

    # (chongyi zheng): vae pickle file
    if checkpoint:
        vae_path = osp.join(checkpoint_dir, 'vae.pkl')
        skewfit_variant['vae_path'] = vae_path if osp.exists(vae_path) else None

    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['vae_kwargs']['n_envs'] = expl_env_num
    train_vae_variant['imsize'] = imsize
    skewfit_variant['imsize'] = imsize
    skewfit_variant['init_camera'] = init_camera
    # (chongyi zheng): add checkpoint variants
    skewfit_variant['checkpoint'] = checkpoint
    skewfit_variant['checkpoint_dir'] = checkpoint_dir
    skewfit_variant['save_snapshot'] = save_snapshot
    skewfit_variant['save_intervals'] = save_intervals
    skewfit_variant['expl_env_num'] = expl_env_num


def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    skewfit_variant = variant['skewfit_variant']
    train_vae_variant = variant['train_vae_variant']

    if skewfit_variant.get('vae_path', None) is None:
        logger.remove_tabular_output('progress.csv', relative_to_snapshot_dir=True)
        logger.add_tabular_output('vae_progress.csv', relative_to_snapshot_dir=True)
        vae, img_data, idx_data = train_vae(train_vae_variant, return_data=True)
        if skewfit_variant.get('save_vae_data', False):
            skewfit_variant['vae_img_data'] = img_data
            skewfit_variant['vae_idx_data'] = idx_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output('vae_progress.csv',
            relative_to_snapshot_dir=True,)
        logger.add_tabular_output('progress.csv',
            relative_to_snapshot_dir=True,)
        skewfit_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if skewfit_variant.get('save_vae_data', False):
            img_data, idx_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs'])
            skewfit_variant['vae_img_data'] = img_data
            skewfit_variant['vae_idx_data'] = idx_data


def train_vae(variant, return_data=False):
    from rlkit.torch.vae.conv_vae import FairVAE
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch

    representation_size = variant["representation_size"]
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn', generate_vae_dataset)
    img_data, idx_data, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs'])

    logger.save_extra_data(info, file_name='extra_data.pkl')
    logger.get_snapshot_dir()

    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    m = FairVAE(representation_size,
        decoder_output_activation=decoder_activation,
        **variant['vae_kwargs'])
    m.to(ptu.device)

    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, img_data, idx_data
    return m


def generate_vae_dataset(variant):
    env_ids = variant['env_ids']
    N = variant.get('N', 10000)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    oracle_dataset_using_set_to_goal = variant.get(
        'oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_and_oracle_policy_data = variant.get('random_and_oracle_policy_data', False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    non_presampled_goal_img_is_garbage = variant.get(
        'non_presampled_goal_img_is_garbage', None)

    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    info = {}

    import gym
    import multiworld
    multiworld.register_all_envs()
    envs = {}
    for env_id in env_ids:
        env = gym.make(env_id[0])
        env = ImageEnv(
            wrapped_env=env,
            env_background=env_id[1],
            imsize=imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
        )
        env.reset()
        envs[env_id[2]] = env

    if random_and_oracle_policy_data:
        policy_file = load_local_or_remote_file(policy_file)
        policy = policy_file['policy']
        policy.to(ptu.device)
    if random_rollout_data:
        from rlkit.exploration_strategies.ou_strategy import OUStrategy
        policy = OUStrategy(envs[0].action_space)

    n_envs = len(envs)
    N_i = int(N / n_envs)
    N = N_i * n_envs
    images = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
    idxs = np.zeros((N, 1), dtype=np.int32)

    for j, (idx, env) in enumerate(envs.items()):
        for i in range(N_i):
            if random_and_oracle_policy_data:
                num_random_steps = int(N * random_and_oracle_policy_data_split)
                if i < num_random_steps:
                    env.reset()
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]
                else:
                    obs = env.reset()
                    policy.reset()
                    for _ in range(n_random_steps):
                        policy_obs = np.hstack((
                            obs['state_observation'],
                            obs['state_desired_goal'],
                        ))
                        action, _ = policy.get_action(policy_obs)
                        obs, _, _, _ = env.step(action)
            elif oracle_dataset_using_set_to_goal:
                goal = env.sample_goal()
                env.set_to_goal(goal)
                obs = env._get_obs()
            elif random_rollout_data:
                if i % n_random_steps == 0:
                    g = dict(state_desired_goal=env.sample_goal_for_rollout())
                    env.set_to_goal(g)
                    policy.reset()
                    # env.reset()
                u = policy.get_action_from_raw_action(
                    env.action_space.sample())
                obs = env.step(u)[0]
            else:
                env.reset()
                for _ in range(n_random_steps):
                    obs = env.step(env.action_space.sample())[0]

            img = obs['image_observation']
            images[j*N_i+i, :] = unormalize_image(img)
            idxs[j*N_i+i, :] = int(idx[-2:])

            if show:
                img_show(img, imsize)

    img_dataset = images.copy()
    idx_dataset = np.eye(n_envs)[idxs.squeeze()].copy()
    return img_dataset, idx_dataset, info

def img_show(img, imsize):
    img = img.reshape(3, imsize, imsize).transpose()
    img = img[::-1]
    plt.imshow(img)

def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file

    checkpoint = variant.get('checkpoint', False)
    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    env_ids = variant["env_ids"]
    reward_params = variant.get("reward_params", dict())
    expl_env_num = variant.get('expl_env_num', 3)
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presampled_goals_path = variant.get('presampled_goals_path', None)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    construct_presampled_goals = variant.get('construct_presampled_goals', False)
    dataset_path = variant.get('dataset_path', None)

    vae = load_local_or_remote_file(vae_path) if type(
        vae_path) is str else vae_path
    oracle_dataset = load_local_or_remote_file(
        dataset_path).item() if dataset_path else None
    if oracle_dataset:
        oracle_dataset = oracle_dataset['state_desired_goal']
        max_len = min(500, len(oracle_dataset))
        oracle_dataset = oracle_dataset[:max_len]

    # sanity check
    n_envs = len(env_ids)
    assert n_envs >= expl_env_num

    import gym
    import multiworld
    multiworld.register_all_envs()
    expl_envs, eval_envs, oracle_imgs = list(), list(), {}

    for i in range(n_envs):
        env_id = env_ids[i]
        env = gym.make(env_id[0])
        env_background = env_id[1]
        env_name = env_id[2]
        env_idx = int(env_name[-2:])
        if env_idx < expl_env_num:
            ot = np.eye(expl_env_num)[env_idx]
        else:
            ot = np.zeros((expl_env_num, ))
        ot = ot.reshape(1, -1)

        if not do_state_exp:
            image_env = ImageEnv(
                wrapped_env=env,
                env_background=env_background,
                imsize=variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
            if presample_goals:
                if presampled_goals_path is None:
                    image_env.non_presampled_goal_img_is_garbage = True
                    vae_env = VAEWrappedEnv(
                        wrapped_env=image_env,
                        vae=vae,
                        env_idx=ot,
                        imsize=image_env.imsize,
                        decode_goals=render,
                        render_goals=render,
                        render_rollouts=render,
                        reward_params=reward_params,
                        **variant.get('vae_wrapped_env_kwargs', {})
                    )
                    if checkpoint:
                        # dummy presampled_goals
                        dummy_presampled_goals = {
                            'image_desired_goal': np.zeros(1000),
                            'desired_goal': np.zeros(1000),
                            'state_desired_goal': np.zeros(1000),
                            'proprio_desired_goal': np.zeros(1000)
                        }
                        presampled_goals = dummy_presampled_goals
                    else:
                        presampled_goals = variant['generate_goal_dataset_fctn'](
                            env=vae_env,
                            env_id=env_id,
                            **variant['goal_generation_kwargs']
                        )  # not sample goals if use checkpoint
                    del vae_env
                else:
                    presampled_goals = load_local_or_remote_file(
                        presampled_goals_path
                    ).item()
                del image_env
                image_env = ImageEnv(
                    wrapped_env=env,
                    env_background=env_background,
                    imsize=variant.get('imsize'),
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    presampled_goals=presampled_goals,
                    construct_presampled_goals=construct_presampled_goals,
                    **variant.get('image_env_kwargs', {})
                )
                vae_env = VAEWrappedEnv(
                    wrapped_env=image_env,
                    vae=vae,
                    env_idx=ot,
                    env_name=env_name,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    presampled_goals=presampled_goals,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                if construct_presampled_goals:
                    print("Constructing all presampled goals")
                else:
                    print("Presampling all goals")
            else:
                del image_env
                image_env = ImageEnv(
                    wrapped_env=env,
                    env_background=env_background,
                    imsize=variant.get('imsize'),
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                )
                vae_env = VAEWrappedEnv(
                    wrapped_env=image_env,
                    vae=vae,
                    env_idx=ot,
                    env_name=env_name,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                if presample_image_goals_only:
                    presampled_goals = variant['generate_goal_dataset_fctn'](
                        image_env=vae_env.wrapped_env,
                        **variant['goal_generation_kwargs']
                    )
                    image_env.set_presampled_goals(presampled_goals)
                    print("Presampling image goals only")
                else:
                    print("Not using presampled goals")


            if oracle_dataset is not None:
                if checkpoint:
                    oracle_imgs[vae_env.vae_env_name] = \
                        np.zeros([oracle_dataset.shape[0], image_env.image_length])
                else:
                    imgs = []
                    for state in oracle_dataset:
                        state = dict(state_desired_goal=state)
                        img = vae_env.wrapped_env.render_desired_goal_image(state).reshape(1, -1)
                        imgs.append(img)
                    imgs = np.concatenate(imgs, axis=0)
                    oracle_imgs[vae_env.vae_env_name] = imgs

            if env_idx < expl_env_num:
                expl_envs.append(vae_env)
            eval_envs.append(vae_env)

    return expl_envs, eval_envs, oracle_imgs


def skewfit_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.GBVAEBuffer import GoalBlockVAEBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import FairVAETrainer

    expl_envs, eval_envs, oracle_imgs = get_envs(variant)
    env = expl_envs[0]

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae

    replay_buffer = GoalBlockVAEBuffer(
        vae=vae,
        envs=expl_envs,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        variant=variant['replay_buffer_kwargs']
    )

    aligned_buffer = GoalBlockVAEBuffer(
        vae=vae,
        envs=expl_envs,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        variant=variant['aligned_buffer_kwargs']
    )

    schedule = variant['online_vae_trainer_kwargs'] \
        .get('c_mmd_schedule', None)
    if schedule is not None:
        schedule = LinearSchedule(schedule[0], schedule[1], schedule[2])
        variant['online_vae_trainer_kwargs']['c_mmd_schedule'] = schedule

    vae_trainer = FairVAETrainer(
        img_dataset=variant['vae_img_data'],
        idx_dataset=variant['vae_idx_data'],
        oracle_dataset=oracle_imgs,
        model=vae,
        envs=expl_envs,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)

    eval_path_collector = GBVAEEnvPathCollector(
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
        envs=eval_envs,
        policy=MakeDeterministic(policy),
        decode_goals=True,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        oracle_dataset=oracle_imgs,
        tsne_kwargs=variant['tsne_kwargs'],
    )

    expl_path_collector = GBVAEEnvPathCollector(
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
        envs=expl_envs,
        policy=policy,
        decode_goals=True,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )


    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_envs=expl_envs,
        evaluation_envs=eval_envs,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        aligned_buffer=aligned_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        **variant['algo_kwargs']
    )

    if variant['custom_goal_sampler'] == 'replay_buffer':
        goal_env = variant.get('custom_goal_env_name', None)
        for env in expl_envs:
            if goal_env is None:
                env.custom_goal_sampler = replay_buffer.buffers[env.vae_env_name].sample_buffer_goals
            else:
                env.custom_goal_sampler = replay_buffer.buffers[goal_env].sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get('checkpoint', False):
        ori_snapshot_dir = logger.get_snapshot_dir()
        ckpt_snapshot_dir = osp.join(ori_snapshot_dir, 'ckpt')
        process = subprocess.Popen(
            "cd {} && rm -rf ckpt.tmp.tar ckpt && tar -xf ckpt.tar".format(
                osp.abspath(ori_snapshot_dir)),
            shell=True, stdout=subprocess.PIPE)
        process.wait()
        logger.set_snapshot_dir(ckpt_snapshot_dir)
        epoch = logger.load_extra_data('epoch.pkl', mode='pickle')
        algorithm.load(epoch)
        logger.set_snapshot_dir(ori_snapshot_dir)
        algorithm.train(epoch + 1,
                        save_snapshot=variant.get('save_snapshot', False),
                        save_intervals=variant.get('save_intervals', 10))
    else:
        algorithm.train(save_snapshot=variant.get('save_snapshot', False),
                        save_intervals=variant.get('save_intervals', 10))


def get_video_save_func(rollout_function, env, policy, variant):
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video
