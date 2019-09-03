# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch.nn as nn

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
from quantlab.indiv.ste_ops import STEActivation

from quantlab.ImageNet.MobileNetv2.mobilenetv2baseline import MobileNetv2Baseline

class MobileNetv2Residuals(MobileNetv2Baseline):
    """MobileNetv2 Convolutional Neural Network."""
    def __init__(self, capacity=1, expansion=6, quant_schemes=None, 
                 quantAct=True, quantActSTENumLevels=None, quantWeights=True, 
                 weightInqSchedule=None, weightInqBits=2, weightInqStrategy="magnitude", 
                 quantSkipFirstLayer=False):
        
        super().__init__(capacity, expansion)
        c0 = 3
        t0 = int(32 * capacity) * 1
        c1 = int(16 * capacity)
        t1 = c1 * expansion
        c2 = int(24 * capacity)
        t2 = c2 * expansion
        c3 = int(32 * capacity)
        t3 = c3 * expansion
        c4 = int(64 * capacity)
        t4 = c4 * expansion
        c5 = int(96 * capacity)
        t5 = c5 * expansion
        c6 = int(160 * capacity)
        t6 = c6 * expansion
        c7 = int(320 * capacity)
        c8 = max(int(1280 * capacity), 1280)
        
        
        def activ(name, nc):
            if quantAct:
                if quantActSTENumLevels != None and quantActSTENumLevels > 0: 
                    return STEActivation(startEpoch=0, 
                                         numLevels=quantActSTENumLevels)
                else:
                    return StochasticActivation(*quant_schemes[name], nc)
            else: 
                assert(quantActSTENumLevels == None or quantActSTENumLevels <= 0)
                return nn.ReLU(inplace=True)
            
        def conv2d(name, ni, no, kernel_size=3, stride=1, padding=1, bias=False):
            if quantWeights:
                if weightInqSchedule == None:
                    return StochasticConv2d(*quant_schemes[name], ni, no, 
                                            kernel_size=kernel_size, stride=stride, 
                                            padding=padding, bias=bias)
                else:
                    return INQConv2d(ni, no, 
                                     kernel_size=kernel_size, stride=stride, 
                                     padding=padding, bias=bias, 
                                     numBits=weightInqBits, strategy=weightInqStrategy)
            else: 
                return nn.Conv2d(ni, no, 
                                 kernel_size=kernel_size, stride=stride, 
                                 padding=padding, bias=bias)
            
        def linear(name, ni, no, bias=False):
            if quantWeights:
                if weightInqSchedule == None:
                    return StochasticLinear(*quant_schemes[name], ni, no, bias=bias)
                else:
                    return INQLinear(ni, no, bias=bias, 
                                     numBits=weightInqBits, strategy=weightInqStrategy)
            else: 
                return nn.Linear(ni, no, bias=bias)
            
        assert(False) # IMPLEMENTATION INCOMPLETE!!!!
            
        # first block
        self.phi01_conv = nn.Conv2d(c0, t0, kernel_size=3, stride=2, padding=1, bias=False)
        self.phi01_bn   = nn.BatchNorm2d(t0)
        self.phi01_act  = nn.ReLU6(inplace=True)
        self.phi02_conv = nn.Conv2d(t0, t0, kernel_size=3, stride=1, padding=1, groups=t0, bias=False)
        self.phi02_bn   = nn.BatchNorm2d(t0)
        self.phi02_act  = nn.ReLU6(inplace=True)
        self.phi03_conv = nn.Conv2d(t0, c1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi03_bn   = nn.BatchNorm2d(c1)
        # second block
        self.phi04_conv = nn.Conv2d(c1, t1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi04_bn   = nn.BatchNorm2d(t1)
        self.phi04_act  = nn.ReLU6(inplace=True)
        self.phi05_conv = nn.Conv2d(t1, t1, kernel_size=3, stride=2, padding=1, groups=t1, bias=False)
        self.phi05_bn   = nn.BatchNorm2d(t1)
        self.phi05_act  = nn.ReLU6(inplace=True)
        self.phi06_conv = nn.Conv2d(t1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi06_bn   = nn.BatchNorm2d(c2)
        self.phi06_act  = StochasticActivation(*quant_schemes['phi06_act'])
        self.phi07_conv = StochasticConv2d(*quant_schemes['phi07_conv'], c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi07_bn   = nn.BatchNorm2d(t2)
        self.phi07_act  = StochasticActivation(*quant_schemes['phi07_act'])
        self.phi08_conv = StochasticConv2d(*quant_schemes['phi08_conv'], t2, t2, kernel_size=3, stride=1, padding=1, groups=t2, bias=False)
        self.phi08_bn   = nn.BatchNorm2d(t2)
        self.phi08_act  = StochasticActivation(*quant_schemes['phi08_act'])
        self.phi09_conv = StochasticConv2d(*quant_schemes['phi09_conv'], t2, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi09_bn   = nn.BatchNorm2d(c2)
        # third block
        self.phi10_conv = nn.Conv2d(c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi10_bn   = nn.BatchNorm2d(t2)
        self.phi10_act  = nn.ReLU6(inplace=True)
        self.phi11_conv = nn.Conv2d(t2, t2, kernel_size=3, stride=2, padding=1, groups=t2, bias=False)
        self.phi11_bn   = nn.BatchNorm2d(t2)
        self.phi11_act  = nn.ReLU6(inplace=True)
        self.phi12_conv = nn.Conv2d(t2, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi12_bn   = nn.BatchNorm2d(c3)
        self.phi12_act  = StochasticActivation(*quant_schemes['phi12_act'])
        self.phi13_conv = StochasticConv2d(*quant_schemes['phi13_conv'], c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi13_bn   = nn.BatchNorm2d(t3)
        self.phi13_act  = StochasticActivation(*quant_schemes['phi13_act'])
        self.phi14_conv = StochasticConv2d(*quant_schemes['phi14_conv'], t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi14_bn   = nn.BatchNorm2d(t3)
        self.phi14_act  = StochasticActivation(*quant_schemes['phi14_act'])
        self.phi15_conv = StochasticConv2d(*quant_schemes['phi15_conv'], t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi15_bn   = nn.BatchNorm2d(c3)
        self.phi15_act  = StochasticActivation(*quant_schemes['phi15_act'])
        self.phi16_conv = StochasticConv2d(*quant_schemes['phi16_conv'], c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi16_bn   = nn.BatchNorm2d(t3)
        self.phi16_act  = StochasticActivation(*quant_schemes['phi16_act'])
        self.phi17_conv = StochasticConv2d(*quant_schemes['phi17_conv'], t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi17_bn   = nn.BatchNorm2d(t3)
        self.phi17_act  = StochasticActivation(*quant_schemes['phi17_act'])
        self.phi18_conv = StochasticConv2d(*quant_schemes['phi18_conv'], t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi18_bn   = nn.BatchNorm2d(c3)
        # fourth block
        self.phi19_conv = nn.Conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi19_bn   = nn.BatchNorm2d(t3)
        self.phi19_act  = nn.ReLU6(inplace=True)
        self.phi20_conv = nn.Conv2d(t3, t3, kernel_size=3, stride=2, padding=1, groups=t3, bias=False)
        self.phi20_bn   = nn.BatchNorm2d(t3)
        self.phi20_act  = nn.ReLU6(inplace=True)
        self.phi21_conv = nn.Conv2d(t3, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi21_bn   = nn.BatchNorm2d(c4)
        self.phi21_act  = StochasticActivation(*quant_schemes['phi21_act'])
        self.phi22_conv = StochasticConv2d(*quant_schemes['phi22_conv'], c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi22_bn   = nn.BatchNorm2d(t4)
        self.phi22_act  = StochasticActivation(*quant_schemes['phi22_act'])
        self.phi23_conv = StochasticConv2d(*quant_schemes['phi23_conv'], t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi23_bn   = nn.BatchNorm2d(t4)
        self.phi23_act  = StochasticActivation(*quant_schemes['phi23_act'])
        self.phi24_conv = StochasticConv2d(*quant_schemes['phi24_conv'], t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi24_bn   = nn.BatchNorm2d(c4)
        self.phi24_act  = StochasticActivation(*quant_schemes['phi24_act'])
        self.phi25_conv = StochasticConv2d(*quant_schemes['phi25_conv'], c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi25_bn   = nn.BatchNorm2d(t4)
        self.phi25_act  = StochasticActivation(*quant_schemes['phi25_act'])
        self.phi26_conv = StochasticConv2d(*quant_schemes['phi26_conv'], t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi26_bn   = nn.BatchNorm2d(t4)
        self.phi26_act  = StochasticActivation(*quant_schemes['phi26_act'])
        self.phi27_conv = StochasticConv2d(*quant_schemes['phi27_conv'], t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi27_bn   = nn.BatchNorm2d(c4)
        self.phi27_act  = StochasticActivation(*quant_schemes['phi27_act'])
        self.phi28_conv = StochasticConv2d(*quant_schemes['phi28_conv'], c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi28_bn   = nn.BatchNorm2d(t4)
        self.phi28_act  = StochasticActivation(*quant_schemes['phi28_act'])
        self.phi29_conv = StochasticConv2d(*quant_schemes['phi29_conv'], t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi29_bn   = nn.BatchNorm2d(t4)
        self.phi29_act  = StochasticActivation(*quant_schemes['phi29_act'])
        self.phi30_conv = StochasticConv2d(*quant_schemes['phi30_conv'], t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi30_bn   = nn.BatchNorm2d(c4)
        # fifth block
        self.phi31_conv = nn.Conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi31_bn   = nn.BatchNorm2d(t4)
        self.phi31_act  = nn.ReLU6(inplace=True)
        self.phi32_conv = nn.Conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi32_bn   = nn.BatchNorm2d(t4)
        self.phi32_act  = nn.ReLU6(inplace=True)
        self.phi33_conv = nn.Conv2d(t4, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi33_bn   = nn.BatchNorm2d(c5)
        self.phi33_act  = StochasticActivation(*quant_schemes['phi33_act'])
        self.phi34_conv = StochasticConv2d(*quant_schemes['phi34_conv'], c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi34_bn   = nn.BatchNorm2d(t5)
        self.phi34_act  = StochasticActivation(*quant_schemes['phi34_act'])
        self.phi35_conv = StochasticConv2d(*quant_schemes['phi35_conv'], t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi35_bn   = nn.BatchNorm2d(t5)
        self.phi35_act  = StochasticActivation(*quant_schemes['phi35_act'])
        self.phi36_conv = StochasticConv2d(*quant_schemes['phi36_conv'], t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi36_bn   = nn.BatchNorm2d(c5)
        self.phi36_act  = StochasticActivation(*quant_schemes['phi36_act'])
        self.phi37_conv = StochasticConv2d(*quant_schemes['phi37_conv'], c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi37_bn   = nn.BatchNorm2d(t5)
        self.phi37_act  = StochasticActivation(*quant_schemes['phi37_act'])
        self.phi38_conv = StochasticConv2d(*quant_schemes['phi38_conv'], t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi38_bn   = nn.BatchNorm2d(t5)
        self.phi38_act  = StochasticActivation(*quant_schemes['phi38_act'])
        self.phi39_conv = StochasticConv2d(*quant_schemes['phi39_conv'], t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi39_bn   = nn.BatchNorm2d(c5)
        # sixth block
        self.phi40_conv = nn.Conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi40_bn   = nn.BatchNorm2d(t5)
        self.phi40_act  = nn.ReLU6(inplace=True)
        self.phi41_conv = nn.Conv2d(t5, t5, kernel_size=3, stride=2, padding=1, groups=t5, bias=False)
        self.phi41_bn   = nn.BatchNorm2d(t5)
        self.phi41_act  = nn.ReLU6(inplace=True)
        self.phi42_conv = nn.Conv2d(t5, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi42_bn   = nn.BatchNorm2d(c6)
        self.phi42_act  = StochasticActivation(*quant_schemes['phi42_act'])
        self.phi43_conv = StochasticConv2d(*quant_schemes['phi43_conv'], c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi43_bn   = nn.BatchNorm2d(t6)
        self.phi43_act  = StochasticActivation(*quant_schemes['phi43_act'])
        self.phi44_conv = StochasticConv2d(*quant_schemes['phi44_conv'], t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi44_bn   = nn.BatchNorm2d(t6)
        self.phi44_act  = StochasticActivation(*quant_schemes['phi44_act'])
        self.phi45_conv = StochasticConv2d(*quant_schemes['phi45_conv'], t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi45_bn   = nn.BatchNorm2d(c6)
        self.phi45_act  = StochasticActivation(*quant_schemes['phi45_act'])
        self.phi46_conv = StochasticConv2d(*quant_schemes['phi46_conv'], c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi46_bn   = nn.BatchNorm2d(t6)
        self.phi46_act  = StochasticActivation(*quant_schemes['phi46_act'])
        self.phi47_conv = StochasticConv2d(*quant_schemes['phi47_conv'], t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi47_bn   = nn.BatchNorm2d(t6)
        self.phi47_act  = StochasticActivation(*quant_schemes['phi47_act'])
        self.phi48_conv = StochasticConv2d(*quant_schemes['phi48_conv'], t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi48_bn   = nn.BatchNorm2d(c6)
        # seventh block
        self.phi49_conv = nn.Conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi49_bn   = nn.BatchNorm2d(t6)
        self.phi49_act  = nn.ReLU6(inplace=True)
        self.phi50_conv = nn.Conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi50_bn   = nn.BatchNorm2d(t6)
        self.phi50_act  = nn.ReLU6(inplace=True)
        self.phi51_conv = nn.Conv2d(t6, c7, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi51_bn   = nn.BatchNorm2d(c7)
        # classifier
        self.phi52_conv = nn.Conv2d(c7, c8, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi52_bn   = nn.BatchNorm2d(c8)
        self.phi52_act  = nn.ReLU6(inplace=True)
        self.phi53_avg  = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.phi53_fc   = nn.Linear(c8, 1000)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
