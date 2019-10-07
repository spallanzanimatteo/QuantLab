# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch.nn as nn

#from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
#from quantlab.indiv.ste_ops import STEActivation

from quantlab.ImageNet.MobileNetv2.mobilenetv2baseline import MobileNetv2Baseline

class MobileNetv2QuantWeight(MobileNetv2Baseline):
    """MobileNetv2 Convolutional Neural Network."""
    def __init__(self, capacity=1, expansion=6, quant_schemes=None, 
                 quantWeights=True, quantAct=True,
                 weightInqSchedule=None, weightInqLevels=None, 
                 weightInqStrategy="magnitude", weightInqQuantInit=None, 
                 quantSkipFirstLayer=False, quantSkipLastLayer=False, 
                 quantDepthwSep=True, pretrained=False):
        
        super().__init__(capacity, expansion)
        assert(quantAct == False)
        
        c0 = 3
        t0 = int(32 * capacity)
        c1 = int(16 * capacity)
        t1 = c1 * expansion
        c2 = int(24 * capacity)
        t2 = c2 * expansion
        c3 = int(32 * capacity)
        t3 = c3 * expansion
        c4 = int(64 * capacity)
        t4 = c4 * expansion
        c5 = int(96 * capacity)
        t5 = c5 * expansion
        c6 = int(160 * capacity)
        t6 = c6 * expansion
        c7 = int(320 * capacity)
        c8 = max(int(1280 * capacity), 1280)
        
        def conv2d(ni, no, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
            if (quantWeights and 
                (quantDepthwSep or 
                 (ni != groups or ni != no))): # not depthw. sep. layer
                assert(weightInqSchedule != None)
                return INQConv2d(ni, no, 
                                 kernel_size=kernel_size, stride=stride, 
                                 padding=padding, groups=groups, bias=bias, 
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy, 
                                 quantInitMethod=weightInqQuantInit)
            else: 
                return nn.Conv2d(ni, no, 
                                 kernel_size=kernel_size, stride=stride, 
                                 padding=padding, groups=groups, bias=bias)
        
        def activ():
            return nn.ReLU6(inplace=True)
            
        # first block
        if quantSkipFirstLayer:
            self.phi01_conv = conv2d(c0, t0, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.phi01_conv = nn.Conv2d(c0, t0, kernel_size=3, stride=2, padding=1, bias=False)
        self.phi01_bn   = nn.BatchNorm2d(t0)
        self.phi01_act  = activ()
        self.phi02_conv = conv2d(t0, t0, kernel_size=3, stride=1, padding=1, groups=t0, bias=False)
        self.phi02_bn   = nn.BatchNorm2d(t0)
        self.phi02_act  = activ()
        self.phi03_conv = conv2d(t0, c1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi03_bn   = nn.BatchNorm2d(c1)
        # second block
        self.phi04_conv = conv2d(c1, t1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi04_bn   = nn.BatchNorm2d(t1)
        self.phi04_act  = activ()
        self.phi05_conv = conv2d(t1, t1, kernel_size=3, stride=2, padding=1, groups=t1, bias=False)
        self.phi05_bn   = nn.BatchNorm2d(t1)
        self.phi05_act  = activ()
        self.phi06_conv = conv2d(t1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi06_bn   = nn.BatchNorm2d(c2)
        self.phi06_act  = activ()
        self.phi07_conv = conv2d(c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi07_bn   = nn.BatchNorm2d(t2)
        self.phi07_act  = activ()
        self.phi08_conv = conv2d(t2, t2, kernel_size=3, stride=1, padding=1, groups=t2, bias=False)
        self.phi08_bn   = nn.BatchNorm2d(t2)
        self.phi08_act  = activ()
        self.phi09_conv = conv2d(t2, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi09_bn   = nn.BatchNorm2d(c2)
        # third block
        self.phi10_conv = conv2d(c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi10_bn   = nn.BatchNorm2d(t2)
        self.phi10_act  = activ()
        self.phi11_conv = conv2d(t2, t2, kernel_size=3, stride=2, padding=1, groups=t2, bias=False)
        self.phi11_bn   = nn.BatchNorm2d(t2)
        self.phi11_act  = activ()
        self.phi12_conv = conv2d(t2, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi12_bn   = nn.BatchNorm2d(c3)
        self.phi12_act  = activ()
        self.phi13_conv = conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi13_bn   = nn.BatchNorm2d(t3)
        self.phi13_act  = activ()
        self.phi14_conv = conv2d(t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi14_bn   = nn.BatchNorm2d(t3)
        self.phi14_act  = activ()
        self.phi15_conv = conv2d(t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi15_bn   = nn.BatchNorm2d(c3)
        self.phi15_act  = activ()
        self.phi16_conv = conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi16_bn   = nn.BatchNorm2d(t3)
        self.phi16_act  = activ()
        self.phi17_conv = conv2d(t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi17_bn   = nn.BatchNorm2d(t3)
        self.phi17_act  = activ()
        self.phi18_conv = conv2d(t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi18_bn   = nn.BatchNorm2d(c3)
        # fourth block
        self.phi19_conv = conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi19_bn   = nn.BatchNorm2d(t3)
        self.phi19_act  = activ()
        self.phi20_conv = conv2d(t3, t3, kernel_size=3, stride=2, padding=1, groups=t3, bias=False)
        self.phi20_bn   = nn.BatchNorm2d(t3)
        self.phi20_act  = activ()
        self.phi21_conv = conv2d(t3, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi21_bn   = nn.BatchNorm2d(c4)
        self.phi21_act  = activ()
        self.phi22_conv = conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi22_bn   = nn.BatchNorm2d(t4)
        self.phi22_act  = activ()
        self.phi23_conv = conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi23_bn   = nn.BatchNorm2d(t4)
        self.phi23_act  = activ()
        self.phi24_conv = conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi24_bn   = nn.BatchNorm2d(c4)
        self.phi24_act  = activ()
        self.phi25_conv = conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi25_bn   = nn.BatchNorm2d(t4)
        self.phi25_act  = activ()
        self.phi26_conv = conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi26_bn   = nn.BatchNorm2d(t4)
        self.phi26_act  = activ()
        self.phi27_conv = conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi27_bn   = nn.BatchNorm2d(c4)
        self.phi27_act  = activ()
        self.phi28_conv = conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi28_bn   = nn.BatchNorm2d(t4)
        self.phi28_act  = activ()
        self.phi29_conv = conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi29_bn   = nn.BatchNorm2d(t4)
        self.phi29_act  = activ()
        self.phi30_conv = conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi30_bn   = nn.BatchNorm2d(c4)
        # fifth block
        self.phi31_conv = conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi31_bn   = nn.BatchNorm2d(t4)
        self.phi31_act  = activ()
        self.phi32_conv = conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi32_bn   = nn.BatchNorm2d(t4)
        self.phi32_act  = activ()
        self.phi33_conv = conv2d(t4, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi33_bn   = nn.BatchNorm2d(c5)
        self.phi33_act  = activ()
        self.phi34_conv = conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi34_bn   = nn.BatchNorm2d(t5)
        self.phi34_act  = activ()
        self.phi35_conv = conv2d(t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi35_bn   = nn.BatchNorm2d(t5)
        self.phi35_act  = activ()
        self.phi36_conv = conv2d(t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi36_bn   = nn.BatchNorm2d(c5)
        self.phi36_act  = activ()
        self.phi37_conv = conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi37_bn   = nn.BatchNorm2d(t5)
        self.phi37_act  = activ()
        self.phi38_conv = conv2d(t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi38_bn   = nn.BatchNorm2d(t5)
        self.phi38_act  = activ()
        self.phi39_conv = conv2d(t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi39_bn   = nn.BatchNorm2d(c5)
        # sixth block
        self.phi40_conv = conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi40_bn   = nn.BatchNorm2d(t5)
        self.phi40_act  = activ()
        self.phi41_conv = conv2d(t5, t5, kernel_size=3, stride=2, padding=1, groups=t5, bias=False)
        self.phi41_bn   = nn.BatchNorm2d(t5)
        self.phi41_act  = activ()
        self.phi42_conv = conv2d(t5, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi42_bn   = nn.BatchNorm2d(c6)
        self.phi42_act  = activ()
        self.phi43_conv = conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi43_bn   = nn.BatchNorm2d(t6)
        self.phi43_act  = activ()
        self.phi44_conv = conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi44_bn   = nn.BatchNorm2d(t6)
        self.phi44_act  = activ()
        self.phi45_conv = conv2d(t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi45_bn   = nn.BatchNorm2d(c6)
        self.phi45_act  = activ()
        self.phi46_conv = conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi46_bn   = nn.BatchNorm2d(t6)
        self.phi46_act  = activ()
        self.phi47_conv = conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi47_bn   = nn.BatchNorm2d(t6)
        self.phi47_act  = activ()
        self.phi48_conv = conv2d(t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi48_bn   = nn.BatchNorm2d(c6)
        # seventh block
        self.phi49_conv = conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi49_bn   = nn.BatchNorm2d(t6)
        self.phi49_act  = activ()
        self.phi50_conv = conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi50_bn   = nn.BatchNorm2d(t6)
        self.phi50_act  = activ()
        self.phi51_conv = conv2d(t6, c7, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi51_bn   = nn.BatchNorm2d(c7)
        # classifier
        self.phi52_conv = conv2d(c7, c8, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi52_bn   = nn.BatchNorm2d(c8)
        self.phi52_act  = activ()
        self.phi53_avg  = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        
        assert(quantSkipLastLayer)
        self.phi53_fc   = nn.Linear(c8, 1000)

        self._initialize_weights()
        
        if pretrained: 
            self.loadPretrainedTorchVision()
        
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule, 
                                               clearOptimStateOnStep=True)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, INQConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, INQLinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def loadPretrainedTorchVision(self):
        import torchvision as tv
        modelRef = tv.models.mobilenet_v2(pretrained=True)
        stateDictRef = modelRef.state_dict()
        remapping = {'features.0.0': 'phi01_conv',
                     'features.0.1': 'phi01_bn',
                     'features.1.conv.0.0': 'phi02_conv',
                     'features.1.conv.0.1': 'phi02_bn',
                     'features.1.conv.1': 'phi03_conv',
                     'features.1.conv.2': 'phi03_bn',
                     }
                     
        for i, layerBlock in enumerate(range(2,17+1)):
            offset = 3*i + 4
            rExt = {'features.%d.conv.0.0' % (layerBlock,) : 'phi%02d_conv' % (offset+0,),
                    'features.%d.conv.0.1' % (layerBlock,) : 'phi%02d_bn'   % (offset+0,),
                    'features.%d.conv.1.0' % (layerBlock,) : 'phi%02d_conv' % (offset+1,),
                    'features.%d.conv.1.1' % (layerBlock,) : 'phi%02d_bn'   % (offset+1,),
                    'features.%d.conv.2'   % (layerBlock,) : 'phi%02d_conv' % (offset+2,),
                    'features.%d.conv.3'   % (layerBlock,) : 'phi%02d_bn'   % (offset+2,),
                    }
            remapping.update(rExt)
        rExt = {'features.18.0': 'phi52_conv', 
                'features.18.1': 'phi52_bn',
                'classifier.1':  'phi53_fc'
                }
        remapping.update(rExt)
        
        stateDictRefMapped = {ksd.replace(kremap, vremap): vsd 
                              for ksd, vsd in stateDictRef.items()
                              for kremap, vremap in remapping.items()
                              if ksd.startswith(kremap)}
        
        missingFields = {k: v 
                         for k,v in self.state_dict().items() 
                         if k not in stateDictRefMapped}
        assert(len([k 
                    for k in missingFields.keys()
                    if not (k.endswith('.sParam') or 
                            k.endswith('.weightFrozen'))
                    ]) == 0) # assert only INQ-specific fields missing
        
        stateDictRefMapped.update(missingFields)
        self.load_state_dict(stateDictRefMapped, strict=True)
    
if __name__ == '__main__':
    model = MobileNetv2QuantWeight(quantAct=False, quantWeights=True, 
                 weightInqSchedule={}, 
                 weightInqLevels=3, 
                 weightInqStrategy="magnitude-SRQ", 
                 weightInqQuantInit='uniform-perCh-l2opt',
                 quantSkipFirstLayer=True,
                 quantSkipLastLayer=True, 
                 pretrained=True)
    









