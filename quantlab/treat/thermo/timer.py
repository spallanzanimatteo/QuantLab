# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

class Constant(object):
    def __init__(self):
        self.decay_factor = 1.

    def update(self, t):
        pass


class Linear(object):
    def __init__(self, t_start, t_decay):
        self.t_start      = t_start
        self.t_decay      = t_decay
        self.decay_factor = 1.

    def update(self, t):
        t_adjusted        = t - self.t_start
        t_relative        = float(min(max(0, t_adjusted), self.t_decay)) / self.t_decay
        self.decay_factor = 1 - t_relative


class Quadratic(object):
    def __init__(self, t_start, t_decay):
        self.t_start      = t_start
        self.t_decay      = t_decay
        self.decay_factor = 1.

    def update(self, t):
        t_adjusted        = t - self.t_start
        t_relative        = float(min(max(0, t_adjusted), self.t_decay)) / self.t_decay
        self.decay_factor = (1. - t_relative) ** 2
