# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli
# large parts of the code taken or adapted from torchvision

import math
import torch
import torch.nn as nn

#from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
#from quantlab.indiv.ste_ops import STEActivation

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, convGen=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convGen(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convGen(planes, planes, kernel_size=3)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, convGen=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convGen(inplanes, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.conv2 = convGen(width, width, kernel_size=3, 
                             stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = convGen(width, planes * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', quant_schemes=None, 
                 quantWeights=True, quantAct=True,
                 weightInqSchedule=None, weightInqBits=None, weightInqLevels=None,
                 weightInqStrategy="magnitude", weightInqQuantInit=None,
                 quantSkipFirstLayer=False, quantSkipLastLayer=False, pretrained=False):
        
        super(ResNet, self).__init__()
        assert(quantAct == False)
        assert(quantSkipFirstLayer)
        assert(quantSkipLastLayer)
        if weightInqBits != None:
            print('warning: weightInqBits deprecated')
            if weightInqBits == 1:
                weightInqLevels = 2
            elif weightInqBits >= 2:
                weightInqLevels = 2**weightInqBits
            else:
                assert(False) 
        
        def convGen(in_planes, out_planes, kernel_size=None, stride=1, 
                    groups=1, dilation=1, firstLayer=False):
            """3x3 convolution with padding"""
        
            if kernel_size == 3:
                padding = dilation
            elif kernel_size == 1:
                padding = 0
            elif kernel_size == 7:
                padding = 3
            else:
                assert(False)
            
            if firstLayer or not(quantWeights): 
                return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=groups, bias=False, dilation=dilation)
            else:
                return INQConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=groups, bias=False, dilation=dilation, 
                                 numLevels=weightInqLevels, strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        
        class BasicBlockWrap(BasicBlock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, convGen=convGen)
        class BottleneckWrap(Bottleneck):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, convGen=convGen)
                
        if arch == 'resnet18':
            block = BasicBlockWrap
            layers = [2, 2, 2, 2]
        elif arch == 'resnet34':
            block = BasicBlockWrap
            layers = [3, 4, 6, 3]
        elif arch == 'resnet50':
            block = BottleneckWrap
            layers = [3, 4, 6, 3]
        elif arch == 'resnet101':
            block = BottleneckWrap
            layers = [3, 4, 23, 3]
        elif arch == 'resnet152':
            block = BottleneckWrap
            layers = [3, 8, 36, 3]
        else:
            assert(False)
        
        self.createNet(block, layers, convGen,
                 num_classes=1000, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)
        
        if pretrained:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(model_urls[arch])
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            missing_keys_nonInq = [s for s in missing_keys if not (s.endswith('.sParam') or s.endswith('.weightFrozen'))]
            assert(len(unexpected_keys) == 0)
            assert(len(missing_keys_nonInq) == 0)
#            if len(missing_keys) > 0:
#                print('load_state_dict -- missing keys:')
#                print(missing_keys)
#            if len(unexpected_keys) > 0:
#                print('load_state_dict -- unexpected keys:')
#                print(unexpected_keys)
                    
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule, 
                                               clearOptimStateOnStep=True)


    def createNet(self, block, layers, convGen, 
                 num_classes=1000, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = convGen(3, self.inplanes, kernel_size=7, stride=2, firstLayer=True)
#        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       convGen=convGen)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       convGen=convGen)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       convGen=convGen)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       convGen=convGen)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, INQConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, convGen=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                convGen(self.inplanes, planes*block.expansion, 
                        kernel_size=1, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, withStats=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if withStats:
            stats = []
            return stats, x

        return x
    

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
    
    
if __name__ == "__main__":
    model = ResNet(arch='resnet18', quantAct=False, weightInqSchedule={}, 
                   quantSkipFirstLayer=True, quantSkipLastLayer=True, 
                   pretrained=True)
    
    loadModel = True
    if loadModel:
    #    path = '../../../ImageNet/logs/exp038/saves/best-backup.ckpt' # BWN
#        path = '../../../ImageNet/logs/exp043/saves/best.ckpt' # TWN
        path = '../../../ImageNet/logs/exp054/saves/best.ckpt' # BWN
        fullState = torch.load(path, map_location='cpu')
        netState = fullState['indiv']['net']
        model.load_state_dict(netState)
    
        import matplotlib.pyplot as plt
        layerNames = list(netState.keys())
        selectedLayers = ['layer4.0.conv1', 
                          'layer2.1.conv2', 
                          'layer1.0.conv2']
    #    selectedLayers = [l + '.weight' for l in selectedLayers]
        selectedLayers = [l + '.weightFrozen' for l in selectedLayers]
        _, axarr = plt.subplots(len(selectedLayers))
        for ax, layerName in zip(axarr, selectedLayers):
            plt.sca(ax)
            plt.hist(netState[layerName].flatten(), 
                     bins=201, range=(-3,3))
            plt.xlim(-3,3)
            plt.title(layerName)
    
    exportONNX = False
    if exportONNX:
        modelFullPrec = ResNet(arch='resnet18', quantAct=False, quantWeights=False, 
                               weightInqSchedule={}, 
                               quantSkipFirstLayer=True, 
                               quantSkipLastLayer=True, 
                               pretrained=True)
        dummyInput = torch.randn(1, 3, 224, 224)
        pbuf = torch.onnx.export(modelFullPrec, dummyInput, 
                                 "export.onnx", verbose=True, 
                                 input_names=['input'], 
                                 output_names=['output'])
        
        
        
        
        
        
        
    
