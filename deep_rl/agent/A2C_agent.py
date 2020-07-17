#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils import tensor


class A2CAgent(BaseAgent):
    def __init__(self, config, network):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = network
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = tensor(self.task.reset())
        self.keys = ['r','m','v','log_prob', 'a', 'ent' ,'ret', 'adv', 's']

    def step(self):
        config = self.config
        storage = Rollout(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            with torch.no_grad():
                prediction = self.network(states)
            # print(prediction[0].shape)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction[0]))
            self.record_online_return(info)
            # rewards = config.reward_normalizer(rewards)
            # storage.add(prediction)
            storage.add({'r': tensor(rewards),
                         'm': tensor(1 - terminals),
                         'a': prediction[0],
                         'log_prob': prediction[1],
                         'ent': prediction[3],
                         'v': prediction[2],
                         's': states})

            states = tensor(next_states)
            self.total_steps += config.num_workers

        self.states = states
        with torch.no_grad():
            prediction = self.network(states)
        storage.compute_returns(prediction[2], config.discount)
        storage.after_fill(self.keys)
        self.train(storage)
        # # storage.add(prediction)
        # storage.add({'a': prediction[0],
        #              'log_prob': prediction[1],
        #              'ent': prediction[3],
        #              'v': prediction[2]})        
        # storage.placeholder()
        # # storage.placeholder()

        # advantages = tensor(np.zeros(config.num_workers))
        # returns = prediction[2].detach()
        # for i in reversed(range(config.rollout_length)):
        #     returns = storage.r[i] + config.discount * storage.m[i] * returns
        #     if not config.use_gae:
        #         advantages = returns - storage.v[i].detach()
        #     else:
        #         td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
        #         advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
        #     storage.adv[i] = advantages.detach()
        #     storage.ret[i] = returns.detach()

        # storage.after_fill(self.keys)


    def train(self, storage):
        config = self.config
        action, returns, advantages, states = storage.return_keys(['a', 'ret', 'adv', 's'])

        _, log_prob, value, entropy = self.network(states, action)

        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

