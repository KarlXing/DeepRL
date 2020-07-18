#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils import tensor
from .a2c_policy import A2CPolicy


class A2CAgent(BaseAgent):
    def __init__(self, config, network, env):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = env
        self.storage = Rollout(config.rollout_length)
        config.optimizer = 'RMSprop'
        config.lr = 1e-4
        self.policy = A2CPolicy(network, config)
        # self.network = network
        # self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = tensor(self.task.reset())
        self.keys = ['a', 'ret', 'adv', 's']
        self.rollout_filled = 0
        self.rollout_length = config.rollout_length
        self.num_workers = config.num_workers
        self.discount = config.discount

    def step(self):
        # config = self.config
        # storage = Rollout(self.rollout_length)
        with torch.no_grad():
            prediction = self.policy.compute_actions(self.states)
            # print(prediction[0].shape)
        next_states, rewards, terminals, info = self.task.step(to_np(prediction[0]))
        self.record_online_return(info)
            # rewards = config.reward_normalizer(rewards)
            # storage.add(prediction)
        self.storage.add({'r': tensor(rewards),
                         'm': tensor(1 - terminals),
                         'a': prediction[0],
                         'v': prediction[2],
                         's': self.states})

        self.total_steps += self.num_workers
        self.rollout_filled += 1
        self.states = tensor(next_states)

        if self.rollout_filled == self.rollout_length:
            with torch.no_grad():
                prediction = self.policy.compute_actions(self.states)
            self.storage.compute_returns(prediction[2], self.discount)
            self.storage.after_fill(self.keys)

            indices = list(range(self.rollout_length * self.num_workers))
            batch = self.sample(self.storage, indices)
            loss = self.policy.learn_on_batch(batch)

            self.rollout_filled = 0
            self.storage.reset()


    def sample(self, storage, indices):
        batch = {}
        for k in self.keys:
            batch[k] = getattr(storage, k)[indices]
        return batch