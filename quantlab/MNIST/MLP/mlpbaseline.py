# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch.nn as nn


# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# Use the following templates for the `net` and `thermostat` configurations:
#
# "net": {
#   "class": "MLPBaseline",
#   "params": {"capacity": 1},
#   "pretrained": null,
#   "loss_function": {
#     "class": "HingeLoss",
#     "params": {"num_classes": 10}
#   }
# }
#
# "thermostat": {
#   "class": "MLPBaseline",
#   "params": {
#     "noise_schemes": {},
#     "bindings":      []
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

    def forward(self, x, withStats=False):
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
        if withStats:
            stats = []
            stats.append(('phi1_fc_w', self.phi1_fc.weight.data))
            stats.append(('phi2_fc_w', self.phi2_fc.weight.data))
            stats.append(('phi3_fc_w', self.phi3_fc.weight.data))
            stats.append(('phi4_fc_w', self.phi4_fc.weight.data))
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
