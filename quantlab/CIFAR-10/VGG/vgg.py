# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import torch.nn as nn

from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
from quantlab.indiv.ste_ops import STEActivation

class VGG(nn.Module):
    """Quantizable VGG."""
    def __init__(self, capacity=1, quant_schemes=None, 
                 quantAct=True, quantActSTENumLevels=None, quantWeights=True, 
                 weightInqSchedule=None, weightInqBits=None, weightInqLevels=None, 
                 weightInqStrategy="magnitude", 
                 quantSkipFirstLayer=False, quantSkipLastLayer=False, 
                 stepEveryEpoch=False, weightInit=None, 
                 rescaleWeights=False, variant=None, weightInqQuantInit=None):
        
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
        
        c0 = 3
        c1 = int(128 * capacity)
        c2 = int(256 * capacity)
        c3 = int(512 * capacity)
        nh = 1024
        
        # convolutional layers
        if quantSkipFirstLayer:
            self.phi1_conv = nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False)
        else:
            self.phi1_conv = conv2d('phi1_conv', c0, c1)
        self.phi1_bn = nn.BatchNorm2d(c1)
        self.phi1_act = activ('phi1_act', c1)
        self.phi2_conv = conv2d('phi2_conv', c1, c1)
        self.phi2_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi2_bn = nn.BatchNorm2d(c1)
        self.phi2_act = activ('phi2_act', c1)
        self.phi3_conv = conv2d('phi3_conv', c1, c2)
        self.phi3_bn = nn.BatchNorm2d(c2)
        self.phi3_act = activ('phi3_act', c2)
        self.phi4_conv = conv2d('phi4_conv', c2, c2)
        self.phi4_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi4_bn = nn.BatchNorm2d(c2)
        self.phi4_act = activ('phi4_act', c2)
        self.phi5_conv = conv2d('phi5_conv', c2, c3)
        self.phi5_bn = nn.BatchNorm2d(c3)
        self.phi5_act = activ('phi5_act', c3)
        self.phi6_conv = conv2d('phi6_conv', c3, c3)
        self.phi6_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.phi6_bn = nn.BatchNorm2d(c3)
        self.phi6_act = activ('phi6_act', c3)
        # dense layers
        if variant == None:
            self.phi7_fc = linear('phi7_fc', c3*4*4, nh)
            self.phi7_bn = nn.BatchNorm1d(nh)
            self.phi7_act = activ('phi7_act', nh)
            self.phi8_fc = linear('phi8_fc', nh, nh)
            self.phi8_bn = nn.BatchNorm1d(nh)
            self.phi8_act = activ('phi8_act', nh)
            if quantSkipLastLayer:
                self.phi9_fc = nn.Linear(nh, 10, bias=False)
                self.phi9_bn = nn.BatchNorm1d(10)
            else:
                self.phi9_fc = linear('phi9_fc', nh, 10)
                self.phi9_bn = nn.BatchNorm1d(10)
        elif variant == 'VGG-Small':
            assert(quantSkipLastLayer)
            self.phi7_fc = nn.Identity()
            self.phi7_bn = nn.Identity()
            self.phi7_act = nn.Identity()
            self.phi8_fc = nn.Identity()
            self.phi8_bn = nn.Identity()
            self.phi8_act = nn.Identity()
            self.phi9_fc = nn.Linear(c3*4*4, 10, bias=True)
            self.phi9_bn = nn.Identity()
        else:
            assert(False)
        
        
        
        # https://unify.id/wp-content/uploads/2018/03/weight_init_BNN.pdf
        def initWeightFunc(m):
            if (isinstance(m, nn.Conv2d) or 
                isinstance(m, INQConv2d) or 
                isinstance(m, StochasticConv2d)):
                w = m.weight.data
                #not initializing bias here...
                if weightInit == None:
                    pass
                elif weightInit == "He": 
                    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
                elif weightInit == "orthogonal":
                    torch.nn.init.orthogonal_(w, gain=1) 
                else:
                    assert(False)
                    
        self.apply(initWeightFunc)
        
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule, 
                                               clearOptimStateOnStep=True, 
                                               stepEveryEpoch=stepEveryEpoch, 
                                               rescaleWeights=rescaleWeights)

    def forward(self, x, withStats=False):
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
#        x = x.reshape(-1, torch.Tensor(list(x.size()[-3:])).to(torch.int32).prod().item())
        x = x.reshape(x.size(0), -1)
        x = self.phi7_fc(x)
        x = self.phi7_bn(x)
        x = self.phi7_act(x)
        x = self.phi8_fc(x)
        x = self.phi8_bn(x)
        x = self.phi8_act(x)
        x = self.phi9_fc(x)
        x = self.phi9_bn(x)
        if withStats:
            stats = []
            stats.append(('phi1_conv_w', self.phi1_conv.weight.data))
            stats.append(('phi3_conv_w', self.phi3_conv.weight.data))
            stats.append(('phi5_conv_w', self.phi5_conv.weight.data))
#            stats.append(('phi7_fc_w', self.phi7_fc.weight.data))
#            stats.append(('phi8_fc_w', self.phi8_fc.weight.data))
#            stats.append(('phi9_fc_w', self.phi9_fc.weight.data))
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
    
# LOAD NETWORK
if __name__ == '__main__':
    model = VGG(quantAct=False, quantWeights=True, 
                weightInqSchedule={'1': 1.0}, quantSkipFirstLayer=True)
#    path = '../../../CIFAR-10/logs/exp048/saves/epoch1050.ckpt'
#    path = '../../../CIFAR-10/logs/exp057/saves/epoch0900.ckpt'
#    path = '../../../CIFAR-10/logs/exp066/saves/epoch1150.ckpt'
#    path = '../../../CIFAR-10/logs/exp069/saves/epoch0100.ckpt'
#    path = '../../../CIFAR-10/logs/exp308/saves/best.ckpt' # TWN with rescaling
#    path = '../../../CIFAR-10/logs/exp071/saves/best.ckpt' # TWN slow latest
#    path = '../../../CIFAR-10/logs/exp273/saves/best.ckpt' # TWN fast latest
    path = '../../../CIFAR-10/logs/exp032/saves/best.ckpt' # TNN
#    path = '../../../CIFAR-10/logs/exp293/saves/best.ckpt' # BNN
    state_dicts = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dicts['indiv']['net'])
    print('non-quant values, layer 3: %8d' % (
            torch.isnan(model.phi3_conv.weightFrozen).sum(dtype=torch.long).item()))
    print('total values, layer 3:     %8d' % (model.phi3_conv.weightFrozen.numel()))
    
    
    import matplotlib.pyplot as plt
    plt.hist(model.phi3_conv.weightFrozen.flatten(), 
             bins=201)
    plt.hist(model.phi3_conv.weight.detach().flatten(), 
             bins=201)
    
    
    #########################################################
    # verification: no information in non-quantized weights
    #########################################################
    verification = False
    if verification:
        quantModules = INQController.getInqModules(model)
        
        #check proper quantization levels
        from matplotlib import pyplot as plt
        plt.hist(quantModules[4].weightFrozen.detach().flatten().numpy(), bins=30)
        
        #remove non-quantized information for test run
        for m in quantModules:
            m.weight.data.zero_()
        state_dicts['indiv']['net'] = model.state_dict()
        torch.save(state_dicts, path.replace('.ckpt', '_verify.ckpt'))
    
