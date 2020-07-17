#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from ..utils import *
import random


class Replay:
    def __init__(self, memory_size, batch_size, drop_prob=0, to_np=True):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.to_np = to_np

    def feed(self, experience):
        if np.random.rand() < self.drop_prob:
            return
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_data = zip(*sampled_data)
        if self.to_np:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def clear(self):
        self.data = []
        self.pos = 0


class SkewedReplay:
    def __init__(self, memory_size, batch_size, criterion):
        self.replay1 = Replay(memory_size // 2, batch_size // 2)
        self.replay2 = Replay(memory_size // 2, batch_size // 2)
        self.criterion = criterion

    def feed(self, experience):
        if self.criterion(experience):
            self.replay1.feed(experience)
        else:
            self.replay2.feed(experience)

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self):
        data1 = self.replay1.sample()
        data2 = self.replay2.sample()
        if data2 is not None:
            data = list(map(lambda x: np.concatenate(x, axis=0), zip(data1, data2)))
        else:
            data = data1
        return data


class PrioritizedReplay:
    def __init__(self, memory_size, batch_size):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_priority = 1

    def feed(self, experience):
        self.tree.add(self.max_priority, experience)

    def feed_batch(self, experience):
            for exp in experience:
                self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.asarray(priorities) / self.tree.total()

        sampled_data = []
        for i in range(batch_size):
            exp = []
            exp.extend(batch[i])
            exp.append(sampling_probabilities[i])
            exp.append(idxs[i])
            sampled_data.append(exp)

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


class AsyncReplay(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    FEED_BATCH = 3
    UPDATE_PRIORITIES = 4

    def __init__(self, memory_size, batch_size, replay_type=Config.DEFAULT_REPLAY):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.cache_len = 2
        self.replay_type = replay_type
        self.start()

    def run(self):
        if self.replay_type == Config.DEFAULT_REPLAY:
            replay = Replay(self.memory_size, self.batch_size)
        elif self.replay_type == Config.PRIORITIZED_REPLAY:
            replay = PrioritizedReplay(self.memory_size, self.batch_size)
        else:
            raise NotImplementedError

        cache = []
        pending_batch = None

        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]: x.share_memory_()
            sample(0)
            sample(1)

        def sample(cur_cache):
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.FEED:
                replay.feed(data)
            elif op == self.FEED_BATCH:
                if not first:
                    pending_batch = data
                else:
                    for transition in data:
                        replay.feed(transition)
            elif op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
                if pending_batch is not None:
                    for transition in pending_batch:
                        replay.feed(transition)
                    pending_batch = None
            elif op == self.UPDATE_PRIORITIES:
                replay.update_priorities(data)
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def feed_batch(self, exps):
        self.pipe.send([self.FEED_BATCH, exps])

    def sample(self):
        self.pipe.send([self.SAMPLE, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def update_priorities(self, info):
        self.pipe.send([self.UPDATE_PRIORITIES, info])

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)

# Rollout for actor-critic methods
class Rollout:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'ret', 'adv', 'log_prob', 'ent']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            getattr(self, k).append(v)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def after_fill(self, keys):
        for k in keys:
            setattr(self, k, torch.cat(getattr(self, k)[:self.size], dim=0))

    # Todo: use_gae
    def compute_returns(self, next_value, discount):
        setattr(self, 'ret', [None] * self.size)
        setattr(self, 'adv', [None] * self.size)
        returns = next_value
        for i in reversed(range(self.size)):
            returns = self.r[i] + discount * self.m[i] * returns
            advantages = returns - self.v[i]
            self.ret[i] = returns
            self.adv[i] = advantages

    def return_keys(self, keys):
        return [getattr(self,key) for key in keys]

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)