from typing import Any
import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start # increases the LR from start to max over the warmup steps
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi)) # uses a cosine decay formula
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)


class VQEntropyLossScheduler:
    '''
    Scheduler for the entropy loss weight in VQVAE, as described in https://arxiv.org/pdf/2310.05737.pdf:
    <<The weight of Lentropy follows an annealing schedule with a 3x higher starting point 
      and linearly decays to a fixed value of 0.1 within 2k steps>>
    '''
    def __init__(self, decay_steps, weight_max, weight_min) -> None:
        self.decay_steps = decay_steps
        self.weight_max = weight_max
        self.weight_min = weight_min

    def __call__(self, steps):
        if steps < self.decay_steps:
            weight = self.weight_max - (self.weight_max - self.weight_min) / self.decay_steps * steps
        else:
            weight = self.weight_min
        return weight
