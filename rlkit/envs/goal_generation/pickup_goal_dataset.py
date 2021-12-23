from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
    get_image_presampled_goals
)
import numpy as np
import cv2
import os.path as osp
import random

from rlkit.util.io import local_path_from_s3_or_local_path


def setup_pickup_image_env(image_env, num_presampled_goals):
    """
    Image env and pickup env will have presampled goals. VAE wrapper should
    encode whatever presampled goal is sampled.
    """
    presampled_goals = get_image_presampled_goals(image_env,
                                                  num_presampled_goals)
    image_env._presampled_goals = presampled_goals
    image_env.num_goals_presampled = \
    presampled_goals[random.choice(list(presampled_goals))].shape[0]


def get_image_presampled_goals_from_vae_env(env, num_presampled_goals,
                                            env_id=None):
    image_env = env.wrapped_env
    return get_image_presampled_goals(image_env, num_presampled_goals)


def get_image_presampled_goals_from_image_env(env, num_presampled_goals,
                                              env_id=None):
    return get_image_presampled_goals(env, num_presampled_goals)


def generate_vae_dataset(variant):
    return generate_vae_dataset_from_params(**variant)


def generate_vae_dataset_from_params(
        env_ids=None,
        N=10000,
        use_cached=True,
        imsize=84,
        num_channels=1,
        show=False,
        init_camera=None,
        dataset_path=None,
        oracle_dataset=False,
):
    from multiworld.core.image_env import ImageEnv, unormalize_image
    import time

    assert oracle_dataset == True

    n_envs = len(env_ids)
    N_i = int(N / n_envs)
    N = N_i * n_envs
    images = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
    idxs = np.zeros((N, 1), dtype=np.int32)

    for j in range(n_envs):
        env_id = env_ids[j]
        env = env_id[0]
        env_background = env_id[1]
        idx = int(env_id[2][-2:])

        if dataset_path is not None:
            filename = local_path_from_s3_or_local_path(dataset_path)
            dataset = np.load(filename)
            np.random.shuffle(dataset)
            N = dataset.shape[0]
        else:
            import gym
            import multiworld
            multiworld.register_all_envs()
            env = gym.make(env)
            env = ImageEnv(
                    wrapped_env=env,
                    env_background=env_background,
                    imsize=imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
            )
            setup_pickup_image_env(env, num_presampled_goals=N_i)
            env.reset()

            for i in range(N_i):
                img = env._presampled_goals['image_desired_goal'][i]
                images[j*N_i+i, :] = unormalize_image(img)
                idxs[j*N_i + i, :] = idx
                if show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    time.sleep(.2)


    img_dataset = images.copy()
    idx_dataset = np.eye(n_envs)[idxs.squeeze()].copy()
    return img_dataset, idx_dataset, {}
