# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch
import torch.nn as nn

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear
from quantlab.indiv.inq_ops import INQController, INQLinear

class MLP(nn.Module):
    """Quantized Multi-Layer Perceptron (both weights and activations)."""
    def __init__(self, capacity, quant_schemes, 
                 quantAct=True, quantWeights=True, 
                 weightInqSchedule=None):
        super().__init__()
        nh = int(2048 * capacity)
        if weightInqSchedule != None:
            weightInqSchedule = {int(k): v for k, v in weightInqSchedule}
        def activ(name, nc):
            if quantAct:
                return StochasticActivation(*quant_scheme[name], nc)
            else:
                return nn.ReLU()
        def linear(name, ni, no, bias=False):
            if quantWeights:
                if weightInqSchedule != None:
                    return INQLinear(ni, no, bias=bias, numBits=2)
                else: 
                    return StochasticLinear(*quant_scheme[name], ni, no, bias=bias)
            else:
                return nn.Linear(ni, no, bias=bias)
        
        self.phi1_fc  = linear('phi1_fc', 28*28, nh, bias=False)
        self.phi1_bn  = nn.BatchNorm1d(nh)
        self.phi1_act = activ('phi1_act', nh)
        self.phi2_fc  = linear('phi2_fc', nh, nh, bias=False)
        self.phi2_bn  = nn.BatchNorm1d(nh)
        self.phi2_act = activ('phi2_act', nh)
        self.phi3_fc  = linear('phi3_fc', nh, nh, bias=False)
        self.phi3_bn  = nn.BatchNorm1d(nh)
        self.phi3_act = activ('phi3_act', nh)
        self.phi4_fc  = linear('phi4_fc', nh, 10, bias=False)
        self.phi4_bn  = nn.BatchNorm1d(10)
        
        #weightInqSchedule={15: 0.5, 22: 0.75, 30: 0.875, 37: 0.9375, 44: 1.0}
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule)

    def forward(self, x, withStats=False):
        stats = []
        x = x.view(-1, 28*28)
        x = self.phi1_fc(x)
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_fc(x)
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_fc(x)
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_fc(x)
        x = self.phi4_bn(x)
        if withStats:
            stats.append(('phi1_fc_w', self.phi1_fc.weight.data))
            stats.append(('phi2_fc_w', self.phi2_fc.weight.data))
            stats.append(('phi3_fc_w', self.phi3_fc.weight.data))
            stats.append(('phi4_fc_w', self.phi4_fc.weight.data))
            return stats, x
        else: 
            return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
