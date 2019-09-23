# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch
import quantlab.indiv as indiv



class ClampWithGradInwards(torch.autograd.Function):
    """Clamps the input, passes the grads for inputs inside or at the
    """
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
    return x - (x - x.round()).detach()

class STEController(indiv.Controller):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules
        
    def step(self, epoch, optimizer=None):
        #step each STE module
        for m in self.modules: 
            m.step(epoch)
                
    @staticmethod
    def getSteModules(net):
        return [m for m in net.modules() if isinstance(m, STEActivation)]

class STEActivation(torch.nn.Module):
    """quantizes activations according to the straight-through estiamtor (STE). 
    Needs a STEController, if startEpoch > 0"""
    def __init__(self, startEpoch=0, numLevels=3, passGradsWhenClamped=False):
        super().__init__()
        self.startEpoch = startEpoch
        self.started = startEpoch <= 0
#        assert(numLevels % 2 == 1 and numLevels >= 3)
        assert(numLevels >= 2)
        self.numLevels = numLevels
        self.passGradsWhenClamped = passGradsWhenClamped
#        self.absMaxValue = torch.nn.Parameter(torch.zeros(1), 
#                                              requires_grad=False)

    def forward(self, x):
#        if not self.started or torch.isnan(self.absMaxValue).item():
#            self.absMaxValue.data[0] = x.abs().max()
            
        if self.started:
#            factor = 1/self.absMaxValue.item() * (self.numLevels // 2)
#            xclamp = clampWithGrad(x, -1, 1)
            if self.passGradsWhenClamped:
#                xclamp = clampWithGrad(x, -1, 1)
                xclamp = clampWithGradInwards(x, -1, 1)
            else:
                xclamp = x.clamp(-1, 1)
            
            y = xclamp
            y = (y + 1)/2 # map from [-1,1] to [0,1]
            y = STERoundFunctional(y*(self.numLevels - 1))/(self.numLevels - 1)
            y = 2*y - 1
#            factorLevels = (self.numLevels // 2)
#            y = STERoundFunctional(xclamp*factorLevels)/factorLevels
        else:
            y = x
        return y
    
    def step(self, epoch):
        if epoch >= self.startEpoch:
            self.started = True
        


if __name__ == "__main__":
    #TESTING
    u = torch.randn(10, requires_grad=True)
    x = u*2
    
    y = STEActivation(numLevels=2)(x)
#    y = STERoundFunctional(x)
#    y = clampWithGradInwards(x, -1, 1)
    
#    L = (y-torch.ones_like(y)*10).norm(2) # pull to 10
    L = y.norm(2) # pull to 0
    L.backward()
    
    
    
    
    



