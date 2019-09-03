# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch 
import torch.nn as nn
import quantlab.indiv as indiv

class INQController(indiv.Controller):
    def __init__(self, modules, schedule, clearOptimStateOnStep=False, stepEveryEpoch=False):
        super().__init__()
        self.modules = modules
        schedule = {int(k): v for k, v in schedule.items()} #parse string keys to ints
        self.schedule = schedule # dictionary mapping epoch to fraction
        self.clearOptimStateOnStep = clearOptimStateOnStep
        self.fraction = 0.0
        self.stepEveryEpoch = stepEveryEpoch
        
    def step(self, epoch, optimizer=None, tensorboardWriter=None):
        
        if epoch in self.schedule.keys():
            self.fraction = self.schedule[epoch]
        elif self.stepEveryEpoch:
            pass
        else:
            return
        
        #log to tensorboard
        if tensorboardWriter != None:
            tensorboardWriter.add_scalar('INQ/fraction', 
                                         self.fraction, global_step=epoch)
        
        #step each INQ module
        for m in self.modules: 
            m.step(self.fraction)

        #clear optimizer state (e.g. Adam's momentum)
        if self.clearOptimStateOnStep and optimizer != None:
            optimizer.state.clear()
                
    @staticmethod
    def getInqModules(net):
        return [m for m in net.modules() if isinstance(m, INQLinear) or isinstance(m, INQConv2d) or isinstance(m, INQConv1d)]
    

def inqStep(fracNew, fracOld, numBits, strategy, s, weight, weightFrozen):
    
    if fracOld == 0.0 and math.isnan(s): #TODO: add or fractOld == None
        #init n_1, n_2 now that we know the weight range
        s = torch.max(torch.abs(weight)).item()
        
    n_1 = math.floor(math.log((4*s)/3, 2))
    n_2 = int(n_1 + 2 - (2**(numBits-1)))
    
    if strategy == "magnitude-SRQ":
        if fracNew == None:
            return fracNew, s
        
#        #get current weights quantized
#        weightAssembled = inqAssembleWeight(weight, weightFrozen)
#        weightAssembled.data = inqQuantize(weightAssembled.data, n_1, n_2)
#        
#        #get number of weights to quantize & find indexes to freeze
#        numWeights = weightFrozen.numel()
#        numFrozen = int(fracNew*numWeights)
#        idxsFreeze = torch.randperm(numWeights)[:numFrozen]
#        
#        #fill new weight tensors
#        weightFrozen.data.fill_(float('NaN'))
#        weightFrozen.data.flatten()[idxsFreeze] = weightAssembled.data.flatten()[idxsFreeze]
#        weight.data.flatten()[idxsFreeze].fill_(0)
        
        #get current weights quantized
        weightFrozen.copy_(inqQuantize(weight, n_1, n_2))
        numUnFreeze = int((1-fracNew)*weight.numel())
        idxsUnFreeze = torch.randperm(weight.numel())[:numUnFreeze]
        weightFrozen.flatten()[idxsUnFreeze] = float('NaN')
    
    else:
         #keep track of quantized fraction to save time
#        if fracNew == fracOld:
#            return # would crash anyway....
        
        #get number of weights to quantize
        prevCount = weightFrozen.numel() - torch.isnan(weightFrozen).sum(dtype=torch.long).item()
        newCount = int(fracNew*weightFrozen.numel())
        
        #find indexes of weights to quant
        if strategy == "magnitude":
            weight[~torch.isnan(weightFrozen)].fill_(0)
            _, idxsSorted = weight.flatten().abs().sort(descending=True)
        elif strategy == "random":
            idxsSorted = torch.randperm(weight.numel())
        else:
            assert(False)
        idxsFreeze = idxsSorted[:newCount-prevCount]
        
        #quantize the weights at these indexes
        weightFrozen.flatten()[idxsFreeze] = inqQuantize(weight.flatten()[idxsFreeze], n_1, n_2)
        
    return fracNew, s

def inqAssembleWeight(weight, weightFrozen):
    weightFrozen = weightFrozen.detach()
    frozen = ~torch.isnan(weightFrozen)
    weightAssembled = torch.zeros_like(weightFrozen)
    weightAssembled[frozen] = weightFrozen[frozen]
    return weightAssembled + torch.isnan(weightFrozen).float()*weight



class INQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 numBits=2, strategy="magnitude"):
        
        super().__init__(in_features, out_features, bias)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = strategy # "magnitude" or "random" or "magnitude-SRQ"
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        self.weightFrozen = nn.Parameter(torch.full_like(self.weight, float('NaN')), 
                                         requires_grad=False)
        self.sParam = nn.Parameter(torch.full((1,), float('NaN')), 
                                   requires_grad=False)
    
    @property    
    def s(self):
        return self.sParam.item()
    @s.setter
    def s(self, value):
        self.sParam[0] = value
    
    def step(self, fraction):
        self.fraction, self.s = inqStep(fraction, self.fraction, 
                                        self.numBits, self.strategy, self.s, 
                                        self.weight.data, 
                                        self.weightFrozen.data)

    def forward(self, input):
        weightAssembled = inqAssembleWeight(self.weight, self.weightFrozen)
        return nn.functional.linear(input, weightAssembled, self.bias)
    
    
class INQConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros', 
                 numBits=2, strategy="magnitude"):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = strategy
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        weightFrozen = torch.full_like(self.weight, float('NaN'), requires_grad=False)
        self.weightFrozen = nn.Parameter(weightFrozen)
        self.sParam = nn.Parameter(torch.full((1,), float('NaN')), requires_grad=False)
    
    @property    
    def s(self):
        return self.sParam.item()
    @s.setter
    def s(self, value):
        self.sParam[0] = value
        
    def step(self, fraction):
        self.fraction, self.s = inqStep(fraction, self.fraction, 
                                        self.numBits, self.strategy, self.s, 
                                        self.weight.data, 
                                        self.weightFrozen.data)

    def forward(self, input):
        weightAssembled = inqAssembleWeight(self.weight, self.weightFrozen)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv1d(
                    nn.functional.pad(input, expanded_padding, mode='circular'),
                    weightAssembled, self.bias, self.stride,
                    (0,), self.dilation, self.groups)
        return nn.functional.conv1d(input, weightAssembled, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
    
class INQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros', 
                 numBits=2, strategy="magnitude"):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = strategy
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        weightFrozen = torch.full_like(self.weight, float('NaN'), requires_grad=False)
        self.weightFrozen = nn.Parameter(weightFrozen)
        self.sParam = nn.Parameter(torch.full((1,), float('NaN')), requires_grad=False)
    
    @property    
    def s(self):
        return self.sParam.item()
    @s.setter
    def s(self, value):
        self.sParam[0] = value
        
    def step(self, fraction):
        self.fraction, self.s = inqStep(fraction, self.fraction, 
                                        self.numBits, self.strategy, self.s, 
                                        self.weight.data, 
                                        self.weightFrozen.data)

    def forward(self, input):
        weightAssembled = inqAssembleWeight(self.weight, self.weightFrozen)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv2d(nn.functional.pad(input, expanded_padding, mode='circular'),
                                        weightAssembled, self.bias, self.stride,
                                        (0,), self.dilation, self.groups)
        return nn.functional.conv2d(input, weightAssembled, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)




def inqQuantize(weight, n_1, n_2):
    """Quantize a single weight using the INQ quantization scheme."""
    import itertools
    
    quantLevelsPos = (2**i for i in range(n_2, n_1+1))
    quantLevelsNeg = (-2**i for i in range(n_2, n_1+1))
    quantLevels = itertools.chain(quantLevelsPos, [0], quantLevelsNeg)
    
    bestQuantLevel = torch.zeros_like(weight)
    minQuantError = torch.full_like(weight, float('inf'))
    
    for ql in quantLevels:
        qerr = (weight-ql).abs()
        mask = qerr < minQuantError
        bestQuantLevel[mask] = ql
        minQuantError[mask] = qerr[mask]
    
    quantizedWeight = bestQuantLevel
    
    return quantizedWeight

if __name__ == '__main__':
    x = torch.linspace(-2,2,100)
    numBits = 2
    s = torch.max(torch.abs(x)).item()
    n_1 = math.floor(math.log((4*s)/3, 2))
    n_2 = int(n_1 + 2 - (2**(numBits-1)))
    x_q = inqQuantize(x, n_1, n_2)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(x.numpy())
    plt.plot(x_q.numpy())


    model = INQLinear(1, 2, bias=False, 
                      numBits=2, strategy="magnitude-SRQ")
#    model = INQConv2d(1, 2, kernel_size=3, bias=False, 
#                      numBits=2, strategy="magnitude-SRQ")

    print(model.weight)
    print(model.weightFrozen)
    model.step(0.5)
    print(model.weight)
    print(model.weightFrozen)

