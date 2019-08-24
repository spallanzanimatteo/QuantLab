
class Controller(object):
    def __init__(self):
        pass
    
    @staticmethod
    def getControllers(net):
        return [v for v in net.__dict__.values() if isinstance(v, Controller)]
