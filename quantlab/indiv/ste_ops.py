# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
from . import Controller

class ClampWithGradInwards(torch.autograd.Function):
    """Clamps the input, passes the grads for inputs inside or at the range limit."""
    @staticmethod
    def forward(ctx, x, low, high):
        ctx.save_for_backward(x, low, high)
        return x.clamp(low.item(), high.item())

    @staticmethod
    def backward(ctx, grad_incoming):
        x, low, high = ctx.saved_tensors
        
        grad_outgoing = grad_incoming.clone()
        grad_outgoing[(x > high)] = 0
        grad_outgoing[(x < low)] = 0
        grad_outgoing[(x == high)*(grad_incoming < 0)] = 0
        grad_outgoing[(x == low )*(grad_incoming > 0)] = 0
        return grad_outgoing, None, None


def clampWithGrad(x, low, high):
    return x - (x - x.clamp(low,high)).detach()

def clampWithGradInwards(x, low, high):
    return ClampWithGradInwards().apply(x, x.new([low]), x.new([high]))

def STERoundFunctional(x):
    #standard STE rounding/quantization (not including the clamping): 
    # 1. forward: quantize x nearest integer values
    # 2. backward: pass gradients through as without quantization
    return x - (x - x.round()).detach()

class STEController(Controller):
    def __init__(self, modules, clearOptimStateOnStart=False):
        super().__init__()
        self.modules = modules
        self.clearOptimStateOnStart = clearOptimStateOnStart
        
    def step(self, epoch, optimizer=None, tensorboardWriter=None):
        #step each STE module
        for m in self.modules: 
            m.step(epoch, self.clearOptimStateOnStart, optimizer)
                
    @staticmethod
    def getSteModules(net):
        return [m for m in net.modules() if isinstance(m, STEActivation)]

class STEActivation(torch.nn.Module):
    """Quantizes activations according to the straight-through estiamtor (STE). 
    Needs a STEController, if startEpoch > 0. 

    startEpoch: first epoch to start quantizing (default: 0). 
    monitorEpoch: In this epoch, keep track of the maximal activation value (absolute value) for range normalization.
        Then (at epoch >= startEpoch), clamp the values to [-max, max], and then do quantization.
        If monitorEpoch is None, max=1 is used."""
    def __init__(self, startEpoch=0, numLevels=3, passGradsWhenClamped=False, monitorEpoch=None):
        super().__init__()
        self.startEpoch = startEpoch
        self.started = startEpoch <= 0

        self.monitorEpoch = monitorEpoch
        self.monitoring = False
        if monitorEpoch is not None:
            self.monitoring = monitorEpoch == 1 # because the epoch starts at epoch 1
            assert(startEpoch > monitorEpoch and monitorEpoch >= 1)

        assert(numLevels >= 2)
        self.numLevels = numLevels
        self.passGradsWhenClamped = passGradsWhenClamped
        self.absMaxValue = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        if self.monitoring:
                self.absMaxValue.data[0] = max(x.abs().max(), self.absMaxValue.item())
            
        if self.started:
            x = x / self.absMaxValue.item() # map from [-max, max] to [-1, 1]
            if self.passGradsWhenClamped:
                xclamp = clampWithGradInwards(x, -1, 1)
            else:
                xclamp = x.clamp(-1, 1)
            
            y = xclamp
            y = (y + 1)/2 # map from [-1,1] to [0,1]
            # scale to [0, numLevels-1], round to nearest int, scale back: 
            y = STERoundFunctional(y*(self.numLevels - 1))/(self.numLevels - 1) 
            y = 2*y - 1 # map from [0,1] to [-1,1]
            y = y * self.absMaxValue.item() # map from [-1, 1] back to [-max, max]
        else:
            y = x
        return y
    
    def step(self, epoch, clearOptimStateOnStart, optimizer):
        if clearOptimStateOnStart and epoch == self.startEpoch:
            optimizer.state.clear()

        if epoch >= self.startEpoch:
            self.started = True

        if self.monitorEpoch is not None and epoch == self.monitorEpoch:
            self.monitoring = True
            self.absMaxValue.data[0] = 0.0
        else:
            self.monitoring = False
        
    @staticmethod
    def getSteModules(net):
        return [m 
                for m in net.modules() 
                if isinstance(m, STEActivation)]

if __name__ == "__main__":
    #TESTING
    u = torch.randn(10, requires_grad=True)
    x = u*2
    
    y = STEActivation(numLevels=3)(x)
#    y = STERoundFunctional(x)
#    y = clampWithGradInwards(x, -1, 1)
#    L = (y-torch.ones_like(y)*10).norm(2) # pull to 10
    L = y.norm(2) # pull to 0
    L.backward()
