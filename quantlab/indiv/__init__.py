# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

class Controller(object):
    def __init__(self):
        pass
    
    def step(self, epoch, optimizer=None, tensorboardWriter=None):
        pass
        
    def step_preTraining(self, *args, **kwargs):
        self.step(*args, **kwargs)
        
    def step_preValidation(self, *args, **kwargs):
        pass
        
    def step_postOptimStep(self, *args, **kwargs):
        pass
    
    @staticmethod
    def getControllers(net):
        return [v for m in net.modules() 
                  for v in m.__dict__.values() 
                     if isinstance(v, Controller)]
