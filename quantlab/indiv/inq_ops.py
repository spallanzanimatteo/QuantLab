# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch 
import torch.nn as nn
import quantlab.nets as nets

class INQController(nets.Controller):
    def __init__(self, modules, schedule, clearOptimStateOnStep=False):
        super().__init__()
        self.modules = modules
        schedule = {int(k): v for k, v in schedule.items()} #parse string keys to ints
        self.schedule = schedule # dictionary mapping epoch to fraction
        self.clearOptimStateOnStep = clearOptimStateOnStep
        
    def step(self, epoch, optimizer=None):
        #check if INQ fraction needs to be adapted
        if epoch not in self.schedule.keys():
            return
        
        #step each INQ module
        fraction = self.schedule[epoch]
        for m in self.modules: 
            m.step(fraction)

        #clear optimizer state (e.g. Adam's momentum)
        if self.clearOptimStateOnStep and optimizer != None:
            optimizer.state.clear()
                
    @staticmethod
    def getInqModules(net):
        return [m for m in net.modules() if isinstance(m, INQLinear) or isinstance(m, INQConv2d) or isinstance(m, INQConv1d)]
    

def inqStep(fracNew, fracOld, numBits, strategy, n_1, n_2, weight, weightFrozen):
     #keep track of quantized fraction to save time
    if fracNew == fracOld:
        return fracNew, n_1, n_2
    if fracOld == 0.0:
        #init n_1, n_2 now that we know the weight range
        s = torch.max(torch.abs(weight)).item()
        n_1 = math.floor(math.log((4*s)/3, 2))
        n_2 = int(n_1 + 2 - (2**(numBits-1)))
    
    #get number of weights to quantize
    prevCount = weightFrozen.numel() - torch.isnan(weightFrozen).sum(dtype=torch.long).item()
    newCount = int(fracNew*weightFrozen.numel())
    
    #find indexes of weights to quant
    if strategy == "magnitude":
        weight[~torch.isnan(weightFrozen)].fill_(0)
        _, idxsSorted = weight.flatten().abs().sort(descending=True)
    elif strategy == "random":
        idxsSorted = torch.randperm(weight.numel())
    idxsFreeze = idxsSorted[:newCount-prevCount]
    
    #quantize the weights at these indexes
    weightFrozen.flatten()[idxsFreeze] = inqQuantize(weight.flatten()[idxsFreeze], n_1, n_2)
    
#    fracNonQuant = torch.isnan(weightFrozen).sum(dtype=torch.long).item()/weightFrozen.numel()
    
    return fracNew, n_1, n_2

def inqAssembleWeight(weight, weightFrozen):
    weightFrozen = weightFrozen.detach()
    frozen = ~torch.isnan(weightFrozen)
    weightAssembled = torch.zeros_like(weightFrozen)
    weightAssembled[frozen] = weightFrozen[frozen]
    return weightAssembled + torch.isnan(weightFrozen).float()*weight



class INQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 numBits=2, reinitOnStep=False):
        
        super().__init__(in_features, out_features, bias)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = "magnitude" # or "random"
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        weightFrozen = torch.full_like(self.weight, float('NaN'), requires_grad=False)
        self.weightFrozen = nn.Parameter(weightFrozen)
        self.reinitOnStep = reinitOnStep
        
    def step(self, fraction):
        fraction, n_1, n_2 = inqStep(fraction, self.fraction, 
                                     self.numBits, self.strategy, 
                                     self.n_1, self.n_2, 
                                     self.weight.data, 
                                     self.weightFrozen.data)
        self.fraction, self.n_1, self.n_2 = fraction, n_1, n_2
        if self.reinitOnStep:
            self.reset_parameters()

    def forward(self, input):
        weightAssembled = inqAssembleWeight(self.weight, self.weightFrozen)
        return nn.functional.linear(input, weightAssembled, self.bias)
    
    
class INQConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros', 
                 numBits=2, reinitOnStep=False):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = "magnitude" # or "random"
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        weightFrozen = torch.full_like(self.weight, float('NaN'), requires_grad=False)
        self.weightFrozen = nn.Parameter(weightFrozen)
        self.reinitOnStep = reinitOnStep
        
    def step(self, fraction):
        fraction, n_1, n_2 = inqStep(fraction, self.fraction, 
                                     self.numBits, self.strategy, 
                                     self.n_1, self.n_2, 
                                     self.weight.data, 
                                     self.weightFrozen.data)
        self.fraction, self.n_1, self.n_2 = fraction, n_1, n_2
        if self.reinitOnStep:
            self.reset_parameters()

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
                 numBits=2, reinitOnStep=False):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        # set INQ parameters
        self.numBits = numBits
        self.strategy = "magnitude" # or "random"
        self.fraction, self.n_1, self.n_2 = 0.0, None, None
        weightFrozen = torch.full_like(self.weight, float('NaN'), requires_grad=False)
        self.weightFrozen = nn.Parameter(weightFrozen)
        self.reinitOnStep = reinitOnStep
        
    def step(self, fraction):
        fraction, n_1, n_2 = inqStep(fraction, self.fraction, 
                                     self.numBits, self.strategy, 
                                     self.n_1, self.n_2, 
                                     self.weight.data, 
                                     self.weightFrozen.data)
        self.fraction, self.n_1, self.n_2 = fraction, n_1, n_2
        if self.reinitOnStep:
            self.reset_parameters()

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
    
#    alpha = 0
#    beta = 2 ** n_2
#    abs_weight = weight.abs()
#    quantizedWeight = torch.empty_like(weight)
#
#    for i in range(n_2, n_1 + 1):
#        selector = (abs_weight >= (alpha + beta) / 2)*(abs_weight <= 3*beta/2)
#        quantizedWeight[selector] = beta*weight[selector].sign()
#        alpha = 2 ** i
#        beta = 2 ** (i + 1)
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
    
    
    
    
    
    
    