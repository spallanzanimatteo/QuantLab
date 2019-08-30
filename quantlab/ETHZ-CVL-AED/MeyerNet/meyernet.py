# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import torch.nn as nn
import math

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d

class MeyerNet(nn.Module):
    """Audio Event Detection quantized Network."""
    def __init__(self, capacityFactor=1.0, version=1, 
                 quantized=True, quant_scheme=None, 
                 quantFirstLast=True, withTwoAct=False, noTimePooling=False):
        super().__init__()
        self.noTimePooling = noTimePooling
        
        def conv1quant(quant_scheme, ni, no, stride=1, padding=1): 
            return StochasticConv2d(*quant_scheme, ni, no, kernel_size=1, 
                                    stride=stride, padding=0, bias=False)
        def conv3quant(quant_scheme, ni, no, stride=1, padding=1): 
            return StochasticConv2d(*quant_scheme, ni, no, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
        def conv1float(quant_scheme, ni, no, stride=1, padding=1): 
            return nn.Conv2d(ni, no, kernel_size=1, 
                             stride=stride, padding=0, bias=False)
        def conv3float(quant_scheme, ni, no, stride=1, padding=1): 
            return nn.Conv2d(ni, no, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        if quantized:
            conv1 = conv1quant
            conv3 = conv3quant
            activ = lambda quant_scheme, nc: StochasticActivation(*quant_scheme, nc)
            if withTwoAct:
                activ2 = lambda nc: nn.ReLU(inplace=True)
            else:
                activ2 = lambda nc: nn.Identity()
            quantScheme = lambda s: quant_scheme[s] 
        else:
            conv1 = conv1float
            conv3 = conv3float
            activ = lambda quant_scheme, nc: nn.ReLU(inplace=True)
            activ2 = lambda nc: nn.Identity()
            quantScheme = lambda s: None
            
        bnorm = lambda nc: nn.BatchNorm2d(nc) 
#        bnorm = lambda nc: nn.Identity() # don't forget to enable/disable bias
        c = lambda v: math.ceil(v*capacityFactor)
        
        c1, c2, c3, c4, c5, c6 = c(64), c(64), c(128), c(128), c(128), c(128)
        if version >= 2: 
            c1 = c(32)

        if quantFirstLast:
            self.phi1_conv = conv3(quantScheme('phi1_conv'), 1, c1)
        else: 
            self.phi1_conv = conv3float(None, 1, c1)
        self.phi1_act2 = activ2(c1)
        self.phi1_bn   = bnorm(c1)
        self.phi1_act  = activ(quantScheme('phi1_act'), c1)
        
        self.phi2_conv = conv3(quantScheme('phi2_conv'), c1, c2, stride=2)
        self.phi2_act2 = activ2(c2)
        self.phi2_bn   = bnorm(c2)
        self.phi2_act  = activ(quantScheme('phi2_act'), c2)
        
        self.phi3_conv = conv3(quantScheme('phi3_conv'), c2, c3)
        self.phi3_act2 = activ2(c3)
        self.phi3_bn   = bnorm(c3)
        self.phi3_act  = activ(quantScheme('phi3_act'), c3)
        
        if version >= 3: 
            self.phi4_do = nn.Dropout2d(0.5)
        else: 
            self.phi4_do   = nn.Identity()
        self.phi4_conv = conv3(quantScheme('phi4_conv'), c3, c4, stride=2)
        self.phi4_act2 = activ2(c4)
        self.phi4_bn   = bnorm(c4)
        self.phi4_act  = activ(quantScheme('phi4_act'), c4)
        
        self.phi5_conv = conv3(quantScheme('phi5_conv'), c4, c5)
        self.phi5_act2 = activ2(c5)
        self.phi5_bn   = bnorm(c5)
        self.phi5_act  = activ(quantScheme('phi5_act'), c5)
        
        self.phi6_conv = conv1(quantScheme('phi6_conv'), c5, c6)
        self.phi6_act2 = activ2(c6)
        self.phi6_bn   = bnorm(c6)
        
        if quantFirstLast:
            self.phi6_act  = activ(quantScheme('phi6_act'), c6)
            self.phi7_conv = conv1(quantScheme('phi7_conv'), c6, 28)
        else:
            self.phi6_act  = nn.Identity()
            self.phi7_conv = conv1float(None, c6, 28)
        self.phi7_bn   = bnorm(28)
        
        if noTimePooling:
            self.phi8_pool = nn.AvgPool2d(kernel_size=(16,1), stride=1, padding=0)
        else:
            self.phi8_pool = nn.AvgPool2d(kernel_size=(16,100), stride=1, padding=0)

    def forward(self, x, withStats=False):
        stats = []
        x = self.phi1_conv(x)
        x = self.phi1_act2(x)
        x = self.phi1_bn(x)
        x = self.phi1_act(x)
        x = self.phi2_conv(x)
        x = self.phi2_act2(x)
        x = self.phi2_bn(x)
        x = self.phi2_act(x)
        x = self.phi3_conv(x)
        x = self.phi3_act2(x)
        x = self.phi3_bn(x)
        x = self.phi3_act(x)
        x = self.phi4_do(x)
        x = self.phi4_conv(x)
        x = self.phi4_act2(x)
        x = self.phi4_bn(x)
        x = self.phi4_act(x)
        x = self.phi5_conv(x)
        x = self.phi5_act2(x)
        x = self.phi5_bn(x)
        x = self.phi5_act(x)
        x = self.phi6_conv(x)
        x = self.phi6_act2(x)
        x = self.phi6_bn(x)
        x = self.phi6_act(x)
        x = self.phi7_conv(x)
        x = self.phi7_bn(x)
        x = self.phi8_pool(x)
        
        if self.noTimePooling:
            x = x.permute(0,2,3,1).reshape(-1, 28)
        else:
            x = x.reshape(x.size(0), 28)
        
        if withStats:
            stats.append(('phi1_conv_w', self.phi1_conv.weight.data))
            stats.append(('phi3_conv_w', self.phi3_conv.weight.data))
            stats.append(('phi5_conv_w', self.phi5_conv.weight.data))
            stats.append(('phi7_conv_w', self.phi7_conv.weight.data))
            return stats, x
        else: 
            return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
