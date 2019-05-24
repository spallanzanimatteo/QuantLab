# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch.nn as nn

from quantlab.nets.stochastic_ops import StochasticActivation, StochasticLinear


############################
## BASELINE ARCHITECTURES ##
############################
#
# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# To configure baseline architectures and their thermostats, use the following
# templates, replacing `topology_name` with the name of the suitable baseline
# and adding the necessary parameters to instantiate the architecture:
#
# "architecture": {
#   "class":  "`topology_name`",
#   "params": {}
# }
#
# "thermostat": {
#   "class":  "`topology_name`",
#   "params": {
#     "noise_scheme": {},
#     "bindings":     []
#   }
# }

class MLPBaseline(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, capacity):
        super(MLPBaseline, self).__init__()
        nh = int(2048 * capacity)
        self.phi1_fc  = nn.Linear(28 * 28, nh, bias=False)
        self.phi1_bn  = nn.BatchNorm1d(nh)
        self.phi1_act = nn.ReLU6()
        self.phi2_fc  = nn.Linear(nh, nh, bias=False)
        self.phi2_bn  = nn.BatchNorm1d(nh)
        self.phi2_act = nn.ReLU6()
        self.phi3_fc  = nn.Linear(nh, nh, bias=False)
        self.phi3_bn  = nn.BatchNorm1d(nh)
        self.phi3_act = nn.ReLU6()
        self.phi4_fc  = nn.Linear(nh, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
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
        return x

    def forward_with_tensor_stats(self, x):
        stats = []
        x = x.view(-1, 28 * 28)
        x = self.phi1_fc(x)
        stats.append(('phi1_fc_w', self.phi1_fc.weight.data))
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_fc(x)
        stats.append(('phi2_fc_w', self.phi2_fc.weight.data))
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_fc(x)
        stats.append(('phi3_fc_w', self.phi3_fc.weight.data))
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_fc(x)
        stats.append(('phi4_fc_w', self.phi4_fc.weight.data))
        return x, stats


#############################
## QUANTIZED ARCHITECTURES ##
#############################

class MLP(nn.Module):
    """Quantized Multi-Layer Perceptron (both weights and activations)."""
    def __init__(self, capacity, quant_scheme):
        super(MLP, self).__init__()
        nh = int(2048 * capacity)
        self.phi1_fc  = StochasticLinear(*quant_scheme['phi1_fc'], 28 * 28, nh, bias=False)
        self.phi1_bn  = nn.BatchNorm1d(nh)
        self.phi1_act = StochasticActivation(*quant_scheme['phi1_act'], nh)
        self.phi2_fc  = StochasticLinear(*quant_scheme['phi2_fc'], nh, nh, bias=False)
        self.phi2_bn  = nn.BatchNorm1d(nh)
        self.phi2_act = StochasticActivation(*quant_scheme['phi2_act'], nh)
        self.phi3_fc  = StochasticLinear(*quant_scheme['phi3_fc'], nh, nh, bias=False)
        self.phi3_bn  = nn.BatchNorm1d(nh)
        self.phi3_act = StochasticActivation(*quant_scheme['phi3_act'], nh)
        self.phi4_fc  = StochasticLinear(*quant_scheme['phi4_fc'], nh, 10, bias=False)
        self.phi4_bn  = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
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
        return x

    def forward_with_tensor_stats(self, x):
        stats = []
        x = x.view(-1, 28 * 28)
        x = self.phi1_fc(x)
        stats.append(('phi1_fc_w', self.phi1_fc.weight.data))
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_fc(x)
        stats.append(('phi2_fc_w', self.phi2_fc.weight.data))
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_fc(x)
        stats.append(('phi3_fc_w', self.phi3_fc.weight.data))
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_fc(x)
        stats.append(('phi4_fc_w', self.phi4_fc.weight.data))
        x = self.phi4_bn(x)
        return x, stats
