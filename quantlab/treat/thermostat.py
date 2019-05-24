# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

from collections import defaultdict
import copy
import numpy as np

from quantlab.nets.stochastic_ops import StochasticActivation, StochasticLinear, _StochasticConvNd
import quantlab.treat.timer as timer


class Thermostat(object):
    """Scheduler of noise distribution through the Neural Network.

    This object is in charge of updating the variances of the stochastic
    processes that describe the Neural Network being trained.
    """

    def __init__(self, net, noise_scheme, bindings, INQ=None):
        self.t            = 0
        self.state        = defaultdict(dict)
        self.noise_scheme = noise_scheme
        self.timers       = list()
        for map_group in bindings:
            # create timers for map group
            t_forward  = getattr(timer, map_group['forward']['class'])(**map_group['forward']['params'])
            t_backward = getattr(timer, map_group['backward']['class'])(**map_group['backward']['params'])
            self.timers.append((t_forward, t_backward))
            maps = list()
            for group_name in map_group['maps']:
                maps = maps + self._solve_maps(net, group_name)
            # expand schedule dictionary to register annealing algorithm
            for m in maps:
                self.noise_scheme[m]['module']       = getattr(net, m)
                num_channels                         = self._get_num_channels(getattr(net, m))
                self.noise_scheme[m]['stddev_start'] = self.noise_scheme[m]['stddev_start'] * np.ones(num_channels)
                self.noise_scheme[m]['forward']      = t_forward
                self.noise_scheme[m]['backward']     = t_backward
        if INQ is not None:
            i_start = 0
            for map_group in INQ["bindings"]:
                maps = list()
                for group_name in map_group['maps']:
                    maps = maps + self._solve_maps(net, group_name)
                map_group_channels = [self._get_num_channels(getattr(net, m)) for m in maps]
                assert all(num_c == map_group_channels[0] for num_c in map_group_channels)
                i_end = i_start + map_group_channels[0]
                for m in maps:
                    self.noise_scheme[m]['INQ_mask_slice'] = np.s_[i_start:i_end]
                i_start = i_end
            self.INQ             = dict()
            self.INQ['schedule'] = INQ['schedule']
            self.INQ['mask']     = np.ones(i_end)
            self.INQ['perm']     = np.random.permutation(i_end)
        else:
            self.INQ = None

    def __getstate__(self):
        return {'t': self.t, 'INQ': self.INQ}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return {'t': self.t, 'INQ': self.INQ}

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

    def _get_num_channels(self, mod):
        if isinstance(mod, StochasticActivation):
            channels_attr = 'num_channels'
        elif isinstance(mod, StochasticLinear):
            channels_attr = 'out_features'
        elif isinstance(mod, _StochasticConvNd):
            channels_attr = 'out_channels'
        return getattr(mod, channels_attr)

    def update_INQ(self):
        p = self.INQ['schedule'][str(self.t)]
        self.INQ['mask'][self.INQ['perm'][0:int(p * len(self.INQ['mask']))]] = 0.

    def step(self):
        self.t = self.t + 1
        for (t_forward, t_backward) in self.timers:
            t_forward.update(self.t)
            t_backward.update(self.t)
        if (self.INQ is not None) and (str(self.t) in self.INQ['schedule'].keys()):
            self.update_INQ()
        for m, scheme in self.noise_scheme.items():
            stddev     = copy.copy(scheme['stddev_start'])
            stddev[0] *= scheme['forward'].decay_factor
            stddev[1] *= scheme['backward'].decay_factor
            if 'INQ_mask_slice' in scheme.keys():
                mask    = self.INQ['mask'][scheme['INQ_mask_slice']]
                stddev *= mask
            scheme['module'].set_stddev(stddev)
