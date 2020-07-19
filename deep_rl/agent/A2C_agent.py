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
        BaseAgent.__init__(self, config, env)
        config.optimizer = 'RMSprop'
        config.lr = 1e-4
        self.policy = A2CPolicy(network, config)
        self.storage = Rollout(config.rollout_length)
        self.rollout_length = config.rollout_length
        self.sample_keys = ['s', 'a', 'ret', 'adv']
        self.rollout_filled = 0

    def step(self):
        with torch.no_grad():
            action, log_prob, v, ent = self.policy.compute_actions(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        self.record_online_return(info)

        self.storage.add({'a': action,
                          'v': v, 
                          'r': tensor(rwd),
                          'm': tensor(1-done),
                          's': self.state})
        self.state = tensor(next_state)
        self.rollout_filled += 1
        # self.total_steps += self.num_workers

        if self.rollout_filled == self.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.policy.compute_actions(self.state)
            self.storage.compute_returns(v, self.discount)
            self.storage.after_fill(self.sample_keys)

            indices = list(range(self.rollout_length*self.num_workers))
            batch = self.sample(indices)
            loss = self.policy.learn_on_batch(batch)

            self.rollout_filled = 0
            self.storage.reset()


    def sample(self, indices):
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[indices]
        return batch