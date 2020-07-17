#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils import tensor


class PPOAgent(BaseAgent):
    def __init__(self, config, network):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = network
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = tensor(self.states)
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 - step / config.max_steps)
        self.keys = ['r','m','v','log_prob', 'a', 'ent' ,'ret', 'adv', 's']

    def step(self):
        config = self.config
        storage = Rollout(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            with torch.no_grad():
                prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction[0]))
            self.record_online_return(info)
            # rewards = config.reward_normalizer(rewards)
            next_states = tensor(next_states)
            # storage.add(prediction)
            storage.add({'a': prediction[0],
                         'log_prob': prediction[1],
                         'v': prediction[2],
                         'ent': prediction[3]})
            storage.add({'r': tensor(rewards),
                         'm': tensor(1 - terminals),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        with torch.no_grad():
            prediction = self.network(states)
        storage.compute_returns(prediction[2], config.discount)
        # storage.add(prediction)
        # storage.add({'a':prediction[0],
        #              'log_prob': prediction[1],
        #              'v': prediction[2],
        #              'ent': prediction[3]})
        # storage.placeholder()

        # advantages = tensor(np.zeros((config.num_workers)))
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

        storage.after_fill(self.keys)
        self.train(storage)



    def train(self, storage):
        config = self.config
        states, actions, log_probs_old, returns, advantages = storage.return_keys(['s', 'a', 'log_prob', 'ret', 'adv'])
        actions = actions
        log_probs_old = log_probs_old
        advantages = (advantages - advantages.mean()) / advantages.std()

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction[1] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction[3].mean()

                value_loss = 0.5 * (sampled_returns - prediction[2]).pow(2).mean()

                approx_kl = (sampled_log_probs_old - prediction[1]).mean()
                if config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()
