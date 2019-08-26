# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

from collections import defaultdict
import copy

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, _StochasticConvNd
import quantlab.treat.thermo.timer as timer


class Thermostat(object):
    """Scheduler of noise distribution through the Neural Network.

    This object is in charge of updating the variances of the stochastic
    processes that describe the Neural Network being trained.
    """
    def __init__(self, net, noise_schemes, bindings):
        self.t             = 0
        self.state         = defaultdict(dict)
        self.noise_schemes = noise_schemes
        self.timers        = list()
        for map_group in bindings:
            # create timers for map group
            t_forward  = getattr(timer, map_group['forward']['class'])(**map_group['forward']['params'])
            t_backward = getattr(timer, map_group['backward']['class'])(**map_group['backward']['params'])
            self.timers.append((t_forward, t_backward))
            # expand schedule dictionary to register annealing algorithm
            maps = list()
            for group_name in map_group['maps']:
                maps = maps + self._solve_maps(net, group_name)
            for m in maps:
                scheme = self.noise_schemes[m]
                scheme['module']       = getattr(net, m)
                scheme['stddev_start'] = scheme['stddev_start']
                scheme['t_forward']    = t_forward
                scheme['t_backward']   = t_backward

    def __getstate__(self):
        return {'t': self.t}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        self.__setstate__({k: v for k, v in state_dict.items()})

    def _is_stoch_mod(self, module):
        _is_stoch_mod = False
        _is_stoch_mod = _is_stoch_mod or isinstance(module, StochasticActivation)
        _is_stoch_mod = _is_stoch_mod or isinstance(module, StochasticLinear)
        _is_stoch_mod = _is_stoch_mod or isinstance(module, _StochasticConvNd)
        return _is_stoch_mod

    def _solve_maps(self, net, map_name):
        return [n for n, mod in net._modules.items() if n.startswith(map_name) and self._is_stoch_mod(mod)]

    def step(self):
        self.t = self.t + 1
        for (t_forward, t_backward) in self.timers:
            t_forward.update(self.t)
            t_backward.update(self.t)
        for m, scheme in self.noise_schemes.items():
            stddev     = copy.copy(scheme['stddev_start'])
            stddev[0] *= scheme['t_forward'].decay_factor
            stddev[1] *= scheme['t_backward'].decay_factor
            scheme['module'].set_stddev(stddev)
