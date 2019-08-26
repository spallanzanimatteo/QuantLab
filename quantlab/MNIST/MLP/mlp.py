# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch.nn as nn

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear


class MLP(nn.Module):
    """Quantized Multi-Layer Perceptron (both weights and activations)."""
    def __init__(self, capacity, quant_schemes):
        super(MLP, self).__init__()
        nh = int(2048 * capacity)
        self.phi1_fc  = StochasticLinear(*quant_schemes['phi1_fc'], 28 * 28, nh, bias=False)
        self.phi1_bn  = nn.BatchNorm1d(nh)
        self.phi1_act = StochasticActivation(*quant_schemes['phi1_act'])
        self.phi2_fc  = StochasticLinear(*quant_schemes['phi2_fc'], nh, nh, bias=False)
        self.phi2_bn  = nn.BatchNorm1d(nh)
        self.phi2_act = StochasticActivation(*quant_schemes['phi2_act'])
        self.phi3_fc  = StochasticLinear(*quant_schemes['phi3_fc'], nh, nh, bias=False)
        self.phi3_bn  = nn.BatchNorm1d(nh)
        self.phi3_act = StochasticActivation(*quant_schemes['phi3_act'])
        self.phi4_fc  = StochasticLinear(*quant_schemes['phi4_fc'], nh, 10, bias=False)
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
        return stats, x
