# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

class Controller(object):
    def __init__(self):
        pass
    
    @staticmethod
    def getControllers(net):
        return [v for m in net.modules() 
                  for v in m.__dict__.values() 
                     if isinstance(v, Controller)]
