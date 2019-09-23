# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import itertools
import torch 
import torch.nn as nn
import quantlab.indiv as indiv



class INQController(indiv.Controller):
    """Instantiate typically once per network, provide it with a list of INQ 
    modules to control and a INQ schedule, and insert a call to the step 
    function once per epoch. """
    def __init__(self, modules, schedule, clearOptimStateOnStep=False, 
                 stepEveryEpoch=False, rescaleWeights=False):
        super().__init__()
        self.modules = modules
        schedule = {int(k): v for k, v in schedule.items()} #parse string keys to ints
        self.schedule = schedule # dictionary mapping epoch to fraction
        self.clearOptimStateOnStep = clearOptimStateOnStep
        self.fraction = 0.0
        self.stepEveryEpoch = stepEveryEpoch
        self.rescaleWeights = rescaleWeights
        
    def step_preTraining(self, epoch, optimizer=None, tensorboardWriter=None):
        
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
               
    def step_postOptimStep(self, *args, **kwargs):
        if self.rescaleWeights:
            #print('rescaling')
            for m in self.modules:
                m.weightInqCtrl.rescaleWeights()
#                m.weight.data.mul_(0.5/m.weight.data.abs().mean().item())
    
    @staticmethod
    def getInqModules(net):
        return [m 
                for m in net.modules() 
                if (isinstance(m, INQLinear) or isinstance(m, INQConv1d) or 
                    isinstance(m, INQConv2d))]

    
class INQParameterController:
    """Used to implement INQ functionality within a custom layer (e.g. INQConv2d).
    Creates and register all relevant fields and parameters in the module. """
    def __init__(self, module, parameterName, numLevels=3, 
                 strategy="magnitude", backCompat=True, 
                 quantInitMethod=None):#'uniform-l1opt'
#        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        self.module = module
        self.parameterName = parameterName
        self.backCompat = backCompat
        
        self.numLevels = numLevels
        self.strategy = strategy # "magnitude" or "random" or "magnitude-SRQ"
        self.fraction = 0.0
        self.quantInitMethod = quantInitMethod
        
        if self.backCompat:
            assert(parameterName == 'weight')
            assert(not hasattr(module, 'weightFrozen'))
            assert(not hasattr(module, 'sParam'))
            self.pnameFrozen = 'weightFrozen'
            self.pnameS = 'sParam'
        else:
            #more structured; adds support for multiple indep. INQ parameters
            self.pnameFrozen = parameterName + '_inqFrozen'
            self.pnameS = parameterName + '_inqS'
            
#        module.register_parameter(pnameFrozen, 
#                                  nn.Parameter(torch.full_like(self.weight, float('NaN')), 
#                                               requires_grad=False))
#        module.register_parameter(pnameS, 
#                                  nn.Parameter(torch.full((1,), float('NaN')), 
#                                               requires_grad=False))
            
        module.__setattr__(self.pnameFrozen, 
                           nn.Parameter(torch.full_like(self.weight, float('NaN')), 
                                        requires_grad=False))
        module.__setattr__(self.pnameS, 
                           nn.Parameter(torch.full((1,), float('NaN')).to(self.weight), 
                                        requires_grad=False))
    
    def getWeightParams(self, module):
        weight = module.__getattr__(self.parameterName)
        weightFrozen = module.__getattr__(self.pnameFrozen)
        return weight, weightFrozen
    
    @property
    def weight(self):
        return self.module.__getattr__(self.parameterName)
    
    @property
    def weightFrozen(self):
        return self.module.__getattr__(self.pnameFrozen)
    
    @property
    def sParam(self):
        return self.module.__getattr__(self.pnameS)
    
    @property
    def s(self):
        return self.sParam.item()
    @s.setter
    def s(self, value):
        self.sParam[0] = value
        
    @staticmethod
    def inqQuantize(weight, quantLevels):
        """Quantize a single weight using the INQ quantization scheme."""
        
        bestQuantLevel = torch.zeros_like(weight)
        minQuantError = torch.full_like(weight, float('inf'))
        
        for ql in quantLevels:
            qerr = (weight-ql).abs()
            mask = qerr < minQuantError
            bestQuantLevel[mask] = ql
            minQuantError[mask] = qerr[mask]
        
        quantizedWeight = bestQuantLevel
        return quantizedWeight
    
    def inqStep(self, fraction):
        
        
        if self.quantInitMethod == None:
            #update s
            if self.fraction == 0.0 and math.isnan(self.s):
                self.s = torch.max(torch.abs(self.weight.data)).item()
            
            #compute quantization levels
            n_1 = math.floor(math.log((4*self.s)/3, 2))
            n_2 = int(n_1 + 2 - (self.numLevels // 2))
            if self.numLevels >= 3:
                quantLevelsPos = (2**i for i in range(n_2, n_1+1))
                quantLevelsNeg = (-2**i for i in range(n_2, n_1+1))
                quantLevels = itertools.chain(quantLevelsPos, [0], quantLevelsNeg)
            else: 
                assert(self.numLevels == 2)
                quantLevels = [self.s/2, -self.s/2]#[2**n_2, -2**n_2]
                
        elif self.quantInitMethod in ['uniform-l1opt', 
                                      'uniform-l2opt', 
                                      'uniform-perCh-l2opt', 
                                      'uniform-linfopt']:
#            import numpy
#            getQLs = lambda s: numpy.linspace(-s, s, num=self.numLevels).tolist()
            getQLs = lambda s: torch.linspace(-s, s, steps=self.numLevels)
            if self.fraction == 0.0 and math.isnan(self.s):
                import scipy.optimize
                def optimWeight(weight):
                    def loss(s):
                        s = s.item()
                        qls = getQLs(s)
                        for i, ql in enumerate(qls):
                            tmp = (weight-ql).abs()
                            if i == 0:
                                minQuantErr = tmp
                            else:
                                minQuantErr = torch.min(minQuantErr, tmp)
                        if self.quantInitMethod == 'uniform-l1opt':
                            return minQuantErr.norm(p=1).item()
                        elif self.quantInitMethod in ['uniform-l2opt', 'uniform-perCh-l2opt']:
                            return minQuantErr.norm(p=2).item()
                        elif self.quantInitMethod == 'uniform-linfopt':
                            return minQuantErr.norm(p=float('inf')).item()
                        else:
                            assert(False)
                    bounds = (1e-6, weight.abs().max().item())
                    optRes = scipy.optimize.brute(loss, ranges=(bounds,), 
                                                  Ns=1000, disp=True, 
                                                  finish=scipy.optimize.fmin)
                    s = optRes[0]
                    weight.mul_(1/s)
                    s = 1 
                    return s
                
                if self.quantInitMethod in ['uniform-l1opt', 
                                            'uniform-l2opt', 
                                            'uniform-linfopt']:
                    self.s = optimWeight(self.weight.data.flatten().detach())
                elif self.quantInitMethod in ['uniform-perCh-l2opt']:
                    self.s = 1
                    for c in range(self.weight.size(0)):
                        optimWeight(self.weight.data[c].flatten().detach())
            quantLevels = getQLs(self.s)
        else:
            assert(False)
        self.fraction = fraction

#        if self.fraction == 0.0 and math.isnan(self.s):
#                self.s = torch.max(torch.abs(self.weight.data)).item()
#        self.fraction = fraction
            
#        #compute quantization levels
#        n_1 = math.floor(math.log((4*self.s)/3, 2))
#        n_2 = int(n_1 + 2 - (self.numLevels // 2))
#        if self.numLevels >= 3:
#            quantLevelsPos = (2**i for i in range(n_2, n_1+1))
#            quantLevelsNeg = (-2**i for i in range(n_2, n_1+1))
#            quantLevels = itertools.chain(quantLevelsPos, [0], quantLevelsNeg)
#        else: 
#            assert(self.numLevels == 2)
#            quantLevels = [self.s/2, -self.s/2]#[2**n_2, -2**n_2]
        
        if self.strategy == "magnitude-SRQ":# or self.strategy == "magnitude-SRQ-perBatch":
            if self.fraction == None:
                return
            
            #get current weights quantized
            self.weightFrozen.data.copy_(self.inqQuantize(self.weight.data, quantLevels))
            numUnFreeze = int((1-self.fraction)*self.weight.numel())
            idxsUnFreeze = torch.randperm(self.weight.numel())[:numUnFreeze]
            self.weightFrozen.data.flatten()[idxsUnFreeze] = float('NaN')
        
        else:
            #get number of weights to quantize
            prevCount = self.weightFrozen.numel() - torch.isnan(self.weightFrozen.data).sum(dtype=torch.long).item()
            newCount = int(self.fraction*self.weightFrozen.numel())
            
            #find indexes of weights to quant
            if self.strategy == "magnitude":
                self.weight.data[~torch.isnan(self.weightFrozen.data)].fill_(0)
                _, idxsSorted = self.weight.data.flatten().abs().sort(descending=True)
            elif self.strategy == "random":
                idxsSorted = torch.randperm(self.weight.numel())
            else:
                assert(False)
            idxsFreeze = idxsSorted[:newCount-prevCount]
            
            #quantize the weights at these indexes
            self.weightFrozen.data.flatten()[idxsFreeze] = self.inqQuantize(self.weight.data.flatten()[idxsFreeze], quantLevels)
    
    def inqAssembleWeight(self, module=None):
        
        #with nn.DataParallel, the module is copied, so self.module cannot be used
        weight, weightFrozen = self.getWeightParams(module)
        
        weightFrozen = weightFrozen.detach()
        frozen = ~torch.isnan(weightFrozen)
        weightAssembled = torch.zeros_like(weightFrozen)
        weightAssembled[frozen] = weightFrozen[frozen]
        fullPrecSelector = torch.isnan(weightFrozen).float()
        tmp = fullPrecSelector*weight
        weightAssembled = weightAssembled + tmp
        return weightAssembled
    
    def rescaleWeights(self):
        self.weight.data.mul_((self.s/2)/self.weight.data.abs().mean().item())


class INQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 numLevels=3, strategy="magnitude", quantInitMethod=None):
        
        super().__init__(in_features, out_features, bias)
        self.weightInqCtrl = INQParameterController(self, 'weight', 
                                                    numLevels, strategy,
                                                    quantInitMethod=quantInitMethod)
    
    def step(self, fraction):
        self.weightInqCtrl.inqStep(fraction)

    def forward(self, input):
        weightAssembled = self.weightInqCtrl.inqAssembleWeight(self)
        return nn.functional.linear(input, weightAssembled, self.bias)
    
    
class INQConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros', 
                 numLevels=3, strategy="magnitude", quantInitMethod=None):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        self.weightInqCtrl = INQParameterController(self, 'weight', 
                                                    numLevels, strategy,
                                                    quantInitMethod=quantInitMethod)
        
    def step(self, fraction):
        self.weightInqCtrl.inqStep(fraction)

    def forward(self, input):
        weightAssembled = self.weightInqCtrl.inqAssembleWeight(self)
        
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
                 numLevels=3, strategy="magnitude", quantInitMethod=None):
        
        super().__init__(in_channels, out_channels, kernel_size, 
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        self.weightInqCtrl = INQParameterController(self, 'weight', 
                                                    numLevels, strategy,
                                                    quantInitMethod=quantInitMethod)
        
    def step(self, fraction):
        self.weightInqCtrl.inqStep(fraction)

    def forward(self, input):
#        if self.strategy == "magnitude-SRQ-perBatch":
#            self.step(self.fraction)
        weightAssembled = self.weightInqCtrl.inqAssembleWeight(self)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv2d(nn.functional.pad(input, expanded_padding, mode='circular'),
                                        weightAssembled, self.bias, self.stride,
                                        (0,), self.dilation, self.groups)

        return nn.functional.conv2d(input, weightAssembled, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

if __name__ == '__main__':
    x = torch.linspace(-2,2,100)
    numLevels = 3
    s = torch.max(torch.abs(x)).item()
    
    n_1 = math.floor(math.log((4*s)/3, 2))
    n_2 = int(n_1 + 2 - (numLevels//2))
    quantLevelsPos = (2**i for i in range(n_2, n_1+1))
    quantLevelsNeg = (-2**i for i in range(n_2, n_1+1))
    quantLevels = itertools.chain(quantLevelsPos, [0], quantLevelsNeg)
    
    x_q = INQParameterController.inqQuantize(x, quantLevels)
    
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(x.numpy())
    plt.plot(x_q.numpy())


    model = INQLinear(2, 3, bias=False, 
                      numLevels=numLevels, strategy="magnitude-SRQ")
#    model = INQConv2d(1, 2, kernel_size=3, bias=False, 
#                      numBits=2, strategy="magnitude-SRQ")

    print(model.weight)
    print(model.weightFrozen)
    model.step(0.5)
    print(model.weight)
    print(model.weightFrozen)
    
    x = torch.randn(4,2)
    y = model(x)
    L = y.norm(p=2)
    L.backward()
    
