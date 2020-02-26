# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

class Controller(object):
    """Abstract class to implement controllers for e.g. quantization layers. 
    Help implementing triggers/behavior based on epoch index, such as starting to quantize or moving. 
    Standard use is to have one or more controllers in the model (found with Controller.getControllers) 
    and call all of the step functions at the corresponding location in the training loop. In turn, a 
    typical controller is responsible for a list of layers such as several (often all) INQConv2d. """
    def __init__(self):
        pass
    
    def step(self, epoch, optimizer=None, tensorboardWriter=None):
        """Alias to step_pretraining."""
        pass
        
    def step_preTraining(self, *args, **kwargs):
        """To be called each epoch"""
        self.step(*args, **kwargs)
        
    def step_preValidation(self, *args, **kwargs):
        pass
        
    def step_postOptimStep(self, *args, **kwargs):
        pass
    
    @staticmethod
    def getControllers(net):
        """Provides a list of all the controllers in the given network."""
        return [v for m in net.modules() 
                  for v in m.__dict__.values() 
                     if isinstance(v, Controller)]
