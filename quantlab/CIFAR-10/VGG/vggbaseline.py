# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn


# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# Use the following templates for the `net` and `thermostat` configurations:
#
# "net": {
#   "class": "VGGBaseline",
#   "params": {"capacity": 1},
#   "pretrained": null,
#   "loss_function": {
#     "class": "HingeLoss",
#     "params": {"num_classes": 10}
#   }
# }
#
# "thermostat": {
#   "class": "VGGBaseline",
#   "params": {
#     "noise_schemes": {},
#     "bindings":      []
#   }
# }

class VGGBaseline(nn.Module):
    """VGG-like Convolutional Neural Network."""
    def __init__(self, capacity):
        super(VGGBaseline, self).__init__()
        c0 = 3
        c1 = int(128 * capacity)
        c2 = int(128 * 2 * capacity)
        c3 = int(128 * 4 * capacity)
        nh = 1024
        # convolutional layers
        self.phi1_conv = nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi1_bn   = nn.BatchNorm2d(c1)
        self.phi1_act  = nn.ReLU6()
        self.phi2_conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi2_mp   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi2_bn   = nn.BatchNorm2d(c1)
        self.phi2_act  = nn.ReLU6()
        self.phi3_conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi3_bn   = nn.BatchNorm2d(c2)
        self.phi3_act  = nn.ReLU6()
        self.phi4_conv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi4_mp   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi4_bn   = nn.BatchNorm2d(c2)
        self.phi4_act  = nn.ReLU6()
        self.phi5_conv = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi5_bn   = nn.BatchNorm2d(c3)
        self.phi5_act  = nn.ReLU6()
        self.phi6_conv = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi6_mp   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi6_bn   = nn.BatchNorm2d(c3)
        self.phi6_act  = nn.ReLU6()
        # fully connected layers
        self.phi7_fc   = nn.Linear(c3 * 4 * 4, nh, bias=False)
        self.phi7_bn   = nn.BatchNorm1d(nh)
        self.phi7_act  = nn.ReLU6()
        self.phi8_fc   = nn.Linear(nh, nh, bias=False)
        self.phi8_bn   = nn.BatchNorm1d(nh)
        self.phi8_act  = nn.ReLU6()
        self.phi9_fc   = nn.Linear(nh, 10)

    def forward(self, x):
        x = self.phi1_conv(x)
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
        x = self.phi4_mp(x)
        x = self.phi4_bn(x)
        x = self.phi4_act(x)
        x = self.phi5_conv(x)
        x = self.phi5_bn(x)
        x = self.phi5_act(x)
        x = self.phi6_conv(x)
        x = self.phi6_mp(x)
        x = self.phi6_bn(x)
        x = self.phi6_act(x)
        x = x.view(-1, torch.Tensor(list(x.size()[-3:])).to(torch.int32).prod().item())
        x = self.phi7_fc(x)
        x = self.phi7_bn(x)
        x = self.phi7_act(x)
        x = self.phi8_fc(x)
        x = self.phi8_bn(x)
        x = self.phi8_act(x)
        x = self.phi9_fc(x)
        return x

    def forward_with_tensor_stats(self, x):
        stats = []
        x = self.phi1_conv(x)
        stats.append(('phi1_conv_w', self.phi1_conv.weight.data))
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_conv(x)
        x = self.phi2_mp(x)
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_conv(x)
        stats.append(('phi3_conv_w', self.phi3_conv.weight.data))
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_conv(x)
        x = self.phi4_mp(x)
        x = self.phi4_bn(x)
        x = self.phi4_act(x)
        x = self.phi5_conv(x)
        stats.append(('phi5_conv_w', self.phi5_conv.weight.data))
        x = self.phi5_bn(x)
        x = self.phi5_act(x)
        x = self.phi6_conv(x)
        x = self.phi6_mp(x)
        x = self.phi6_bn(x)
        x = self.phi6_act(x)
        x = x.view(-1, torch.Tensor(list(x.size()[-3:])).to(torch.int32).prod().item())
        x = self.phi7_fc(x)
        stats.append(('phi7_fc_w', self.phi7_fc.weight.data))
        x = self.phi7_bn(x)
        x = self.phi7_act(x)
        x = self.phi8_fc(x)
        stats.append(('phi8_fc_w', self.phi8_fc.weight.data))
        x = self.phi8_bn(x)
        x = self.phi8_act(x)
        x = self.phi9_fc(x)
        stats.append(('phi9_fc_w', self.phi9_fc.weight.data))
        return stats, x
