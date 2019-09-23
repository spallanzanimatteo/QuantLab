# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import torch.nn as nn

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
from quantlab.indiv.ste_ops import STEActivation


class AlexNet(nn.Module):
    """Quantized AlexNet (both weights and activations)."""
    def __init__(self, capacity=1, quant_schemes=None, 
                 quantAct=True, quantActSTENumLevels=None, quantWeights=True, 
                 weightInqSchedule=None, weightInqBits=None, weightInqLevels=None, 
                 weightInqStrategy="magnitude", 
                 quantSkipFirstLayer=False, quantSkipLastLayer=False, 
                 withDropout=False, alternateSizes=False, weightInqQuantInit=None):
        
        super().__init__()
        
        assert(weightInqBits == None or weightInqLevels == None)
        if weightInqBits != None:
            print('warning: weightInqBits deprecated')
            if weightInqBits == 1:
                weightInqLevels = 2
            elif weightInqBits >= 2:
                weightInqLevels = 2**weightInqBits
            else:
                assert(False)
                
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
                                     numLevels=weightInqLevels, 
                                     strategy=weightInqStrategy, 
                                     quantInitMethod=weightInqQuantInit)
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
                                     numLevels=weightInqLevels, 
                                     strategy=weightInqStrategy, 
                                     quantInitMethod=weightInqQuantInit)
            else: 
                return nn.Linear(ni, no, bias=bias)
        
        def dropout(p=0.5):
            if withDropout:
                return nn.Dropout(p)
            else:
                return nn.Identity()
            
        if alternateSizes:
            #following LQ-net
            c0 = 3
            c1 = int(96 * capacity)
            c2 = int(256 * capacity)
            c3 = int(384 * capacity)
            c4 = int(384 * capacity)
            c5 = 256
            nh = 4096
        else: 
            c0 = 3
            c1 = int(64 * capacity)
            c2 = int(192 * capacity)
            c3 = int(384 * capacity)
            c4 = int(256 * capacity)
            c5 = 256
            nh = 4096
            
            
        # convolutional layers
        if quantSkipFirstLayer:
            self.phi1_conv = nn.Conv2d(c0, c1, kernel_size=11, 
                                       stride=4, padding=2, bias=False)
        else:
            self.phi1_conv = conv2d('phi1_conv', c0, c1, kernel_size=11, 
                                    stride=4, padding=2, bias=False)
        self.phi1_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi1_bn   = nn.BatchNorm2d(c1)
        self.phi1_act  = activ('phi1_act', c1)
        self.phi2_conv = conv2d('phi2_conv', c1, c2, kernel_size=5, padding=2, bias=False)
        self.phi2_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi2_bn   = nn.BatchNorm2d(c2)
        self.phi2_act  = activ('phi2_act', c2)
        self.phi3_conv = conv2d('phi3_conv', c2, c3, kernel_size=3, padding=1, bias=False)
        self.phi3_bn   = nn.BatchNorm2d(c3)
        self.phi3_act  = activ('phi3_act', c3)
        self.phi4_conv = conv2d('phi4_conv', c3, c4, kernel_size=3, padding=1, bias=False)
        self.phi4_bn   = nn.BatchNorm2d(c4)
        self.phi4_act  = activ('phi4_act', c4)
        self.phi5_conv = conv2d('phi5_conv', c4, c5, kernel_size=3, padding=1, bias=False)
        self.phi5_mp   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.phi5_bn   = nn.BatchNorm2d(c5)
        self.phi5_act  = activ('phi5_act', c5)
        # fully connected layers
        self.phi6_do   = dropout()
        self.phi6_fc   = linear('phi6_fc', c5*6*6, nh, bias=False)
        self.phi6_bn   = nn.BatchNorm1d(nh)
        self.phi6_act  = activ('phi6_act', nh)
        self.phi7_do   = dropout()
        self.phi7_fc   = linear('phi7_fc', nh, nh, bias=False)
        self.phi7_bn   = nn.BatchNorm1d(nh)
        self.phi7_act  = activ('phi7_act', nh)
        
        if quantSkipLastLayer:
            self.phi8_fc   = nn.Linear(nh, 1000, bias=False)
        else:
            self.phi8_fc   = linear('phi8_fc', nh, 1000, bias=False)
        self.phi8_bn   = nn.BatchNorm1d(1000)
        
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule, 
                                               clearOptimStateOnStep=True)

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
        x = self.phi6_do(x)
        x = self.phi6_fc(x)
        x = self.phi6_bn(x)
        x = self.phi6_act(x)
        x = self.phi7_do(x)
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




if __name__ == '__main__':
    model = AlexNet(quantAct=False, quantWeights=True, 
                 weightInqSchedule={}, weightInqBits=2, 
                 weightInqStrategy="magnitude-SRQ", 
                 quantSkipFirstLayer=True)
    
    import torchvision as tv
    modelRef = tv.models.alexnet(pretrained=True)
    stateDictRef = modelRef.state_dict()
    #batch normalization not in original model...?!























    

