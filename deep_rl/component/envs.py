#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# from baselines.common.atari_wrappers import FrameStack as FrameStack_
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env import VecEnvWrapper

from ..utils import *

try:
    import roboschool
except ImportError:
    pass


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
# def make_env(env_id, seed, rank, episode_life=True):
#     def _thunk():
#         random_seed(seed)
#         if env_id.startswith("dm"):
#             import dm_control2gym
#             _, domain, task = env_id.split('-')
#             env = dm_control2gym.make(domain_name=domain, task_name=task)
#         else:
#             env = gym.make(env_id)
#         is_atari = hasattr(gym.envs, 'atari') and isinstance(
#             env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
#         if is_atari:
#             env = make_atari(env_id)
#         env.seed(seed + rank)
#         env = OriginalReturnWrapper(env)
#         if is_atari:
#             env = wrap_deepmind(env,
#                                 episode_life=episode_life,
#                                 clip_rewards=False,
#                                 frame_stack=False,
#                                 scale=False)
#             obs_shape = env.observation_space.shape
#             if len(obs_shape) == 3:
#                 env = TransposeImage(env)
#             env = FrameStack(env, 4)

#         return env

#     return _thunk

def make_env(env_id, seed, rank, custom_wrapper=None):
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)

        if is_atari:
            env = wrap_deepmind(env)

        if custom_wrapper:
            env = custom_wrapper(env)

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env)

        return env

    return _thunk

class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# # The original LayzeFrames doesn't work well
# class LazyFrames(object):
#     def __init__(self, frames):
#         """This object ensures that common frames between the observations are only stored once.
#         It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
#         buffers.

#         This object should only be converted to numpy array before being passed to the model.

#         You'd not believe how complex the previous solution was."""
#         self._frames = frames

#     def __array__(self, dtype=None):
#         out = np.concatenate(self._frames, axis=0)
#         if dtype is not None:
#             out = out.astype(dtype)
#         return out

#     def __len__(self):
#         return len(self.__array__())

#     def __getitem__(self, i):
#         return self.__array__()[i]


# class FrameStack(FrameStack_):
#     def __init__(self, env, k):
#         FrameStack_.__init__(self, env, k)

#     def _get_ob(self):
#         assert len(self.frames) == self.k
#         return LazyFrames(list(self.frames))


# # The original one in baselines is really bad
# class DummyVecEnv(VecEnv):
#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
#         self.actions = None

#     def step_async(self, actions):
#         self.actions = actions

#     def step_wait(self):
#         data = []
#         for i in range(self.num_envs):
#             obs, rew, done, info = self.envs[i].step(self.actions[i])
#             if done:
#                 obs = self.envs[i].reset()
#             data.append([obs, rew, done, info])
#         obs, rew, done, info = zip(*data)
#         return obs, np.asarray(rew), np.asarray(done), info

#     def reset(self):
#         return [env.reset() for env in self.envs]

#     def close(self):
#         return

class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()


def make_vec_envs(env_name, num_workers, seed=1, num_frame_stack=1):
    envs = [make_env(env_name, seed, i) for i in range(num_workers)]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecFrameStack(envs, num_frame_stack)

    return envs


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        # envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        # if single_process:
        #     Wrapper = DummyVecEnv
        # else:
        #     Wrapper = SubprocVecEnv
        # self.env = Wrapper(envs)
        self.env = make_vec_envs(name, num_envs, num_frame_stack=4)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        # if isinstance(self.action_space, Box):
        #     actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
