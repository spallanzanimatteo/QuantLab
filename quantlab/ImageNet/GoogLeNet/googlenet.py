# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli
# large parts of the code taken or adapted from torchvision

import warnings
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#from quantlab.indiv.stochastic_ops import StochasticActivation, StochasticLinear, StochasticConv2d
from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
#from quantlab.indiv.ste_ops import STEActivation


model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, quantized=True, **kwargs):
        super(BasicConv2d, self).__init__()
        if quantized:
            self.conv = INQConv2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 numLevels=3, strategy="magnitude", quantInitMethod=None):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1, 
                                   numLevels=numLevels, strategy=strategy, 
                                   quantInitMethod=quantInitMethod)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1, 
                        numLevels=numLevels, strategy=strategy, 
                        quantInitMethod=quantInitMethod),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1, 
                        numLevels=numLevels, strategy=strategy, 
                        quantInitMethod=quantInitMethod)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1, 
                        numLevels=numLevels, strategy=strategy, 
                        quantInitMethod=quantInitMethod),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1, 
                        numLevels=numLevels, strategy=strategy, 
                        quantInitMethod=quantInitMethod)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1, 
                        numLevels=numLevels, strategy=strategy, 
                        quantInitMethod=quantInitMethod)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, quant_schemes=None, 
                 quantWeights=True, quantAct=True,
                 weightInqSchedule=None, weightInqLevels=None,
                 weightInqStrategy="magnitude", weightInqQuantInit=None,
                 quantSkipFirstLayer=False, quantSkipLastLayer=False, pretrained=False):
        super().__init__()
        assert(quantAct == False)
        assert(quantSkipFirstLayer)
        assert(quantSkipLastLayer)

        self.conv1 = BasicConv2d(3, 64, quantized=False, 
                                 kernel_size=7, stride=2, padding=3)
        
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128,
                                 numLevels=weightInqLevels, 
                                 strategy=weightInqStrategy,
                                 quantInitMethod=weightInqQuantInit)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()
        
        
        if pretrained:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(model_urls['googlenet'])
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            #filter out expected mismatches 
            #(missing auxiliary outputs in model, missing INQ params in pretrained data)
            missing_keys_nonInq = [s for s in missing_keys 
                                   if not (s.endswith('.sParam') or 
                                           s.endswith('.weightFrozen'))]
            unexpected_keys_nonAux = [s for s in unexpected_keys 
                                      if not s.startswith('aux')]

            assert(len(unexpected_keys_nonAux) == 0)
            assert(len(missing_keys_nonInq) == 0)
        
        if weightInqSchedule != None: 
            self.inqController = INQController(INQController.getInqModules(self), 
                                               weightInqSchedule, 
                                               clearOptimStateOnStep=True)

    def _initialize_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or 
                isinstance(m, INQConv2d) or 
                isinstance(m, nn.Linear)):
                
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, withStats=False):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        
        if withStats:
            stats = []
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x


if __name__ == "__main__":
    model = GoogLeNet(quantAct=False, weightInqSchedule={}, 
                   quantSkipFirstLayer=True, quantSkipLastLayer=True, 
                   pretrained=True)
    
    loadModel = False
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
        modelFullPrec = GoogLeNet(quantAct=False, quantWeights=False, 
                               weightInqSchedule={}, 
                               quantSkipFirstLayer=True, 
                               quantSkipLastLayer=True, 
                               pretrained=True)
        dummyInput = torch.randn(1, 3, 224, 224)
        pbuf = torch.onnx.export(modelFullPrec, dummyInput, 
                                 "export.onnx", verbose=True, 
                                 input_names=['input'], 
                                 output_names=['output'])
