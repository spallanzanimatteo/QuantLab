# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import torch.nn as nn


# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# Use the following templates for the `net` and `thermostat` configurations:
#
# "net": {
#   "class": "AlexNetBaseline",
#   "params": {"capacity": 1},
#   "pretrained": null,
#   "loss_fn": {
#     "class": "CrossEntropyLoss",
#     "params": {}
#   }
# }
#
# "thermostat": {
#   "class": "AlexNetBaseline",
#   "params": {
#     "noise_schemes": {},
#     "bindings":      []
#   }
# }

class AlexNetBaseline(nn.Module):
    """AlexNet Convolutional Neural Network."""
    def __init__(self, capacity):
        super().__init__()
        c0 = 3
        c1 = int(64 * capacity)
        c2 = int(64 * 3 * capacity)
        c3 = int(64 * 6 * capacity)
        c4 = int(64 * 4 * capacity)
        c5 = 256
        nh = 4096
        # convolutional layers
        self.phi1_conv = nn.Conv2d(c0, c1, kernel_size=11, stride=4, padding=2, bias=False)
        self.phi1_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi1_bn   = nn.BatchNorm2d(c1)
        self.phi1_act  = nn.ReLU6()
        self.phi2_conv = nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False)
        self.phi2_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi2_bn   = nn.BatchNorm2d(c2)
        self.phi2_act  = nn.ReLU6()
        self.phi3_conv = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.phi3_bn   = nn.BatchNorm2d(c3)
        self.phi3_act  = nn.ReLU6()
        self.phi4_conv = nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False)
        self.phi4_bn   = nn.BatchNorm2d(c4)
        self.phi4_act  = nn.ReLU6()
        self.phi5_conv = nn.Conv2d(c4, c5, kernel_size=3, padding=1, bias=False)
        self.phi5_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi5_bn   = nn.BatchNorm2d(c5)
        self.phi5_act  = nn.ReLU6()
        # fully connected layers
        self.phi6_fc   = nn.Linear(c5 * 6 * 6, nh, bias=False)
        self.phi6_bn   = nn.BatchNorm1d(nh)
        self.phi6_act  = nn.ReLU6()
        self.phi7_fc   = nn.Linear(nh, nh, bias=False)
        self.phi7_bn   = nn.BatchNorm1d(nh)
        self.phi7_act  = nn.ReLU6()
        self.phi8_fc   = nn.Linear(nh, 1000)

    def forward(self, x, withStats=False):
        x = self.phi1_conv(x)
        x = self.phi1_mp(x)
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_conv(x)
        x = self.phi2_mp(x)
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_conv(x)
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_conv(x)
        x = self.phi4_bn(x)
        x = self.phi4_act(x)
        x = self.phi5_conv(x)
        x = self.phi5_mp(x)
        x = self.phi5_bn(x)
        x = self.phi5_act(x)
        x = x.view(-1, torch.Tensor(list(x.size()[-3:])).to(torch.int32).prod().item())
        x = self.phi6_fc(x)
        x = self.phi6_bn(x)
        x = self.phi6_act(x)
        x = self.phi7_fc(x)
        x = self.phi7_bn(x)
        x = self.phi7_act(x)
        x = self.phi8_fc(x)
        x = self.phi8_bn(x)
        if withStats:
            stats = []
            stats.append(('phi1_conv_w', self.phi1_conv.weight.data))
            stats.append(('phi2_conv_w', self.phi2_conv.weight.data))
            stats.append(('phi3_conv_w', self.phi3_conv.weight.data))
            stats.append(('phi4_conv_w', self.phi4_conv.weight.data))
            stats.append(('phi5_conv_w', self.phi5_conv.weight.data))
            stats.append(('phi6_fc_w', self.phi6_fc.weight.data))
            stats.append(('phi7_fc_w', self.phi7_fc.weight.data))
            stats.append(('phi8_fc_w', self.phi8_fc.weight.data))
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
