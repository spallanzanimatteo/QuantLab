# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import math
# from scipy.stats import norm, uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class UniformHeavisideProcess(torch.autograd.Function):
    """A Stochastic Process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and uniform noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x, q, t, s, is_p, training):
        # is_p = isinstance(x, nn.Parameter)
        is_p      = torch.Tensor([is_p]).to(torch.float32)
        ctx.save_for_backward(x, q, t, s, is_p)
        t_shape   = [*t.size()] + [1 for dim in range(x.dim())]  # dimensions with size 1 enable broadcasting
        x_minus_t = x - t.reshape(t_shape)
        if training:
            no_noise = (s[0] == 0.).float()  # channels with zero noise
            sf       = s[0] + no_noise       # to avoid division by zero
            sf_inv   = 1 / sf
            if is_p:
                s_shape = [*s[0].size()] + [1 for dim in range(x.dim() - 1)]  #   Cx(N_1xN_2...xN_n)
            else:
                s_shape = [*s[0].size()] + [1 for dim in range(x.dim() - 2)]  # BxCx(N_1xN_2...xN_n)
            no_noise = no_noise.reshape(s_shape)
            cdf = (1 - no_noise) * torch.clamp((0.5 * x_minus_t) * sf_inv.reshape(s_shape) + 0.5, 0., 1.)\
                      + no_noise * (x_minus_t >= 0.).float()
        else:
            cdf = (x_minus_t >= 0.).float()
        d       = q[1:] - q[:-1]
        sigma_x = q[0] + torch.sum(d.reshape(t_shape) * cdf, 0)
        return sigma_x

    @staticmethod
    def backward(ctx, grad_incoming):
        x, q, t, s, is_p = ctx.saved_tensors
        t_shape   = [*t.size()] + [1 for dim in range(x.dim())]  # dimensions with size 1 enable broadcasting
        x_minus_t = x - t.reshape(t_shape)
        no_noise  = (s[1] == 0.).float()  # channels with zero noise
        sb        = s[1] + no_noise       # to avoid division by zero
        sb_inv    = 1 / sb
        if is_p:
            s_shape = [*s[1].size()] + [1 for dim in range(x.dim()-1)]
        else:
            s_shape = [*s[1].size()] + [1 for dim in range(x.dim()-2)]
        no_noise = no_noise.reshape(s_shape)
        pdf = (1 - no_noise) * (torch.abs_(x_minus_t) <= sb.reshape(s_shape)).float() * (0.5 * sb_inv.reshape(s_shape))
        d              = q[1:] - q[:-1]
        local_jacobian = torch.sum(d.reshape(t_shape) * pdf, 0)
        grad_outgoing  = grad_incoming * local_jacobian
        return grad_outgoing, None, None, None, None, None


# class GaussianHeavisideProcess(torch.autograd.Function):
#     """A Stochastic Process composed by step functions.
#
#     This class defines a stochastic process whose elementary events are step
#     functions with fixed quantization levels (codominion) and Gaussian noise
#     on the jumps positions.
#     """
#     @staticmethod
#     def forward(ctx, x, q, t, s, training):
#         ctx.save_for_backward(x, q, t, s)
#         d = q[1:] - q[:-1]
#         shape = [*t.size(), *torch.ones(x.dim())]  # dimensions with size 1 enable broadcasting
#         if s[0] == 0. or not training:
#             cdf = (x - t.reshape(shape) >= 0.).to(x)
#         else:
#             cdf = torch.Tensor(norm.cdf(x - t.reshape(shape), scale=s[0])).to(x)
#         sigma_x = q[0] + torch.sum(d.reshape(shape) * cdf, 0)
#         return sigma_x
#
#     @staticmethod
#     def backward(ctx, grad_incoming):
#         x, q, t, s = ctx.saved_tensors
#         d = q[1:] - q[:-1]
#         shape = [*t.size(), *torch.ones(x.dim())]  # dimensions with size 1 enable broadcasting
#         if s[1] == 0.:
#             pdf = torch.zeros_like(grad_incoming)
#         else:
#             pdf = torch.Tensor(norm.pdf((x - t.reshape(shape)), scale=s[1])).to(x)
#         local_jacobian = torch.sum(d.reshape(shape) * pdf, 0)
#         grad_outgoing = grad_incoming * local_jacobian
#         return grad_outgoing, None, None, None, None


class StochasticActivation(nn.Module):
    """Quantize scores."""
    def __init__(self, process, quant_levels, thresholds,
                 num_channels):
        super(StochasticActivation, self).__init__()
        self.process = process
        if self.process == 'uniform':
            self.activate = UniformHeavisideProcess.apply
        # elif self.process == 'gaussian':
        #     self.activate = GaussianHeavisideProcess.apply
        super(StochasticActivation, self).register_parameter('quant_levels',
                                                             nn.Parameter(torch.Tensor(quant_levels),
                                                                          requires_grad=False))
        super(StochasticActivation, self).register_parameter('thresholds',
                                                             nn.Parameter(torch.Tensor(thresholds),
                                                                          requires_grad=False))
        stddev = torch.ones(2, num_channels)
        super(StochasticActivation, self).register_parameter('stddev',
                                                             nn.Parameter(torch.Tensor(stddev),
                                                                          requires_grad=False))
        self.num_channels = num_channels

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)

    def forward(self, x):
        return self.activate(x, self.quant_levels, self.thresholds, self.stddev, 0, self.training)


class StochasticLinear(nn.Module):
    """Affine transform with quantized parameters."""
    def __init__(self, process, quant_levels, thresholds,
                 in_features, out_features, bias=True):
        super(StochasticLinear, self).__init__()
        # set stochastic properties
        self.process = process
        if self.process == 'uniform':
            self.activate_weight = UniformHeavisideProcess.apply
        # elif self.process == 'gaussian':
        #     self.activate_weight = GaussianHeavisideProcess.apply
        super(StochasticLinear, self).register_parameter('quant_levels',
                                                         nn.Parameter(torch.Tensor(quant_levels),
                                                                      requires_grad=False))
        super(StochasticLinear, self).register_parameter('thresholds',
                                                         nn.Parameter(torch.Tensor(thresholds),
                                                                      requires_grad=False))
        stddev = torch.ones(2, out_features)
        super(StochasticLinear, self).register_parameter('stddev',
                                                         nn.Parameter(torch.Tensor(stddev),
                                                                      requires_grad=False))
        # set linear layer properties
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # init weights near thresholds
        self.weight.data.random_(to=len(self.thresholds.data))
        self.weight.data = self.thresholds[self.weight.data.to(torch.long)]
        self.weight.data = torch.add(self.weight.data, torch.zeros_like(self.weight.data).uniform_(-stdv, stdv))
        # init biases
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.quant_levels, self.thresholds, self.stddev, 1, self.training)
        return F.linear(input, weight, self.bias)


class _StochasticConvNd(nn.Module):
    """Cross-correlation transform with quantized parameters."""
    def __init__(self, process, quant_levels, thresholds,
                 in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias):
        super(_StochasticConvNd, self).__init__()
        # set stochastic properties
        self.process = process
        if self.process == 'uniform':
            self.activate_weight = UniformHeavisideProcess.apply
        # elif self.process == 'gaussian':
        #     self.activate_weight = GaussianHeavisideProcess.apply
        super(_StochasticConvNd, self).register_parameter('quant_levels',
                                                          nn.Parameter(torch.Tensor(quant_levels),
                                                                       requires_grad=False))
        super(_StochasticConvNd, self).register_parameter('thresholds',
                                                          nn.Parameter(torch.Tensor(thresholds),
                                                                       requires_grad=False))
        stddev = torch.ones(2, out_channels)
        super(_StochasticConvNd, self).register_parameter('stddev',
                                                          nn.Parameter(torch.Tensor(stddev),
                                                                       requires_grad=False))
        # set convolutional layer properties
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.transposed     = transposed
        self.output_padding = output_padding
        self.groups         = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # init weights near thresholds
        self.weight.data.random_(to=len(self.thresholds.data))
        self.weight.data = self.thresholds[self.weight.data.to(torch.long)]
        self.weight.data = torch.add(self.weight.data, torch.zeros_like(self.weight.data).uniform_(-stdv, stdv))
        # init biases
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)


class StochasticConv1d(_StochasticConvNd):
    def __init__(self, process, quant_levels, thresholds,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride      = _single(stride)
        padding     = _single(padding)
        dilation    = _single(dilation)
        super(StochasticConv1d, self).__init__(
              process, quant_levels, thresholds,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.quant_levels, self.thresholds, self.stddev, 1, self.training)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class StochasticConv2d(_StochasticConvNd):
    def __init__(self, process, quant_levels, thresholds,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)
        super(StochasticConv2d, self).__init__(
              process, quant_levels, thresholds,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.quant_levels, self.thresholds, self.stddev, 1, self.training)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class StochasticConv3d(_StochasticConvNd):
    def __init__(self, process, quant_levels, thresholds,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride      = _triple(stride)
        padding     = _triple(padding)
        dilation    = _triple(dilation)
        super(StochasticConv3d, self).__init__(
              process, quant_levels, thresholds,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.quant_levels, self.thresholds, self.stddev, 1, self.training)
        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
