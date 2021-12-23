import os
import os.path as osp

os.chdir(osp.join(os.path.dirname(osp.abspath(__file__)), '../../'))

import multiworld.envs.mujoco as mwmj
import rlkit.util.hyperparameter as hyp
from rlkit.envs.goal_generation.pickup_goal_dataset import (
    generate_vae_dataset,
    get_image_presampled_goals_from_vae_env,
)

import rlkit.torch.vae.vae_schedules as vae_schedules
import rlkit.torch.vae.vae_sampler_schedule as vae_sampler
from multiworld.envs.mujoco.cameras import (
    sawyer_pick_and_place_camera,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture


def setting(idx):
    # presample_goals, presample_path, goal_env_idx, expl_mode, eval_mode
    path = osp.join(osp.dirname(mwmj.__file__),
                    "goals", "pickup_goals.npy",)
    if idx == 1:
        return True, path, 0, 'vae_prior', 'presampled'
    elif idx == 2:
        return True, path, 0, 'custom_goal_sampler', 'presampled'
    elif idx == 3:
        return True, path, None, 'custom_goal_sampler', 'presampled'
    elif idx == 4:
        return True, None, None, 'vae_prior', 'env'
    elif idx == 5:
        return True, None, None, 'custom_goal_sampler', 'env'
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--setting', type=int, default=5)
    parser.add_argument('--mmd', type=float, default=100.0)
    parser.add_argument('--diff', type=float, default=0.04)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--exp-token', type=str, default='-mmd-constant')
    parser.add_argument('--para-token', type=str, default='-debug')
    parser.add_argument('--checkpoint', action='store_true', default=False)
    parser.add_argument('--checkpoint-prefix', type=str, default='./data')
    parser.add_argument('--save-snapshot', action='store_true', default=False)
    parser.add_argument('--save-intervals', type=int, default=10)
    args = parser.parse_args()

    # Define training and evaluation envs
    # Each env_id in env_ids contain [env, env_background, env_unique_token]
    presample, path, goal_idx, expl_mode, eval_mode = setting(args.setting)
    env, version = 'SawyerPickupEnvYZEasy', '-v0'
    env_ids = [
        # training envs
        # same table textures (domains) since it is a sideview
        ['SawyerPickupEnvYZEasy-v0', 'Background-0', 'id-00'],  # cyan
        ['SawyerPickupEnvYZEasy-v0', 'Background-1', 'id-01'],  # yellow
        ['SawyerPickupEnvYZEasy-v0', 'Background-2', 'id-02'],  # pink
        # evaluation envs
        ['SawyerPickupEnvYZEasy-v0', 'Background-4', 'id-03'],
        ['SawyerPickupEnvYZEasy-v0', 'Background-6', 'id-04'],
        ['SawyerPickupEnvYZEasy-v0', 'Background-9', 'id-05'],
    ]

    variant = dict(
        algorithm='PASF',
        imsize=48,
        double_algo=False,
        expl_env_num=3,
        env_ids=env_ids,
        seed=args.seed,
        save_snapshot=args.save_snapshot,
        save_intervals=args.save_intervals,
        skewfit_variant=dict(
            save_video=False,
            save_video_period=50,
            presample_goals=presample,
            presampled_goals_path=path,
            construct_presampled_goals=presample,
            custom_goal_sampler='replay_buffer',
            custom_goal_env_name=None,
            online_vae_trainer_kwargs=dict(
                beta=args.beta,
                lr=1e-3,
                D=1024,
                c_mmd=args.mmd,
                c_diff=args.diff
            ),
            generate_goal_dataset_fctn=get_image_presampled_goals_from_vae_env,
            goal_generation_kwargs=dict(num_presampled_goals=300),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                batch_size=1200,
                num_epochs=500,
                max_path_length=50,
                align_max_path_length=50,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=250,
                num_trains_per_train_loop=1500,
                num_aligned_steps_per_train_loop=50,
                min_num_steps_before_training=0,
                align_min_num_steps_before_training=3000,
                vae_training_schedule=vae_schedules.custom_schedule,
                vae_sampler_schedule=vae_sampler.init_align_then_random,
                oracle_data=False,
                vae_save_period=50,
                use_aligned_buffer=True,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                reward_scale=1,
                discount=0.99,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(40000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode=expl_mode,
            ),
            aligned_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(12000),
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
            ),
            exploration_goal_sampling_mode=expl_mode,
            evaluation_goal_sampling_mode=eval_mode,
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            dataset_path=osp.join(osp.dirname(mwmj.__file__),
                                  "goals", "pickup_goals.npy", ),
            tsne_kwargs=dict(
                use_tsne=True,
                save_period=50,
            )
        ),
        train_vae_variant=dict(
            representation_size=20,
            beta=30,
            num_epochs=0,
            dump_skew_debug_plots=True,
            decoder_activation='gaussian',
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            generate_vae_data_fctn=generate_vae_dataset,
            generate_vae_dataset_kwargs=dict(
                N=30,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
            ),
            algo_kwargs=dict(),
            save_period=10,
        ),
        init_camera=sawyer_pick_and_place_camera,
    )

    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    algo = variant['algorithm']
    seed = variant['seed']

    n_seeds = 1
    mode = 'local'
    exp_prefix = algo + '-pickup-setting-' + str(args.setting) + args.exp_token + args.para_token
    checkpoint_dir = osp.join(args.checkpoint_prefix, exp_prefix, 's-' + str(args.seed))

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                skewfit_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                seed=seed,
                variant=variant,
                checkpoint=args.checkpoint,
                checkpoint_dir=checkpoint_dir,
                base_log_dir=args.checkpoint_prefix,
                use_gpu=True,
                num_exps_per_instance=3,
            )
