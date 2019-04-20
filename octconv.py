import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class OctConv2d(nn.modules.conv._ConvNd):
    """Unofficial implementation of the Octave Convolution in the "Drop an Octave" paper.

    oct_type (str): The type of OctConv you'd like to use. ['first', 'A'] both stand for the the first Octave Convolution.
                    ['last', 'C'] both stand for th last Octave Convolution. And 'regular' stand for the regular ones.
    """
    
    def __init__(self, oct_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, alpha_in=0.5, alpha_out=0.5):
        
        if oct_type not in ('regular', 'first', 'last', 'A', 'C'):
            raise InvalidOctType("Invalid oct_type was chosen!")

        oct_type_dict = {'first': (0, alpha_out), 'A': (0, alpha_out), 'last': (alpha_in, 0), 'C': (alpha_in, 0), 
                         'regular': (alpha_in, alpha_out)}        

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        # TODO: Make it work with any padding
        padding = _pair(int((kernel_size[0] - 1) / 2))
        # padding = _pair(padding)
        dilation = _pair(dilation)
        super(OctConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), 1, bias)

        # Get alphas from the oct_type_dict
        self.oct_type = oct_type
        self.alpha_in, self.alpha_out = oct_type_dict[self.oct_type]
        
        self.num_high_in_channels = int((1 - self.alpha_in) * in_channels)
        self.num_low_in_channels = int(self.alpha_in * in_channels)
        self.num_high_out_channels = int((1 - self.alpha_out) * out_channels)
        self.num_low_out_channels = int(self.alpha_out * out_channels)

        self.high_hh_weight = self.weight[:self.num_high_out_channels, :self.num_high_in_channels, :, :].clone()
        self.high_hh_bias = self.bias[:self.num_high_out_channels].clone()

        self.high_hl_weight = self.weight[self.num_high_out_channels:, :self.num_high_in_channels, :, :].clone()
        self.high_hl_bias = self.bias[self.num_high_out_channels:].clone()

        self.low_lh_weight = self.weight[:self.num_high_out_channels, self.num_high_in_channels:, :, :].clone()
        self.low_lh_bias = self.bias[:self.num_high_out_channels].clone()

        self.low_ll_weight = self.weight[self.num_high_out_channels:, self.num_high_in_channels:, :, :].clone()
        self.low_ll_bias = self.bias[self.num_high_out_channels:].clone()

        self.high_hh_weight.data, self.high_hl_weight.data, self.low_lh_weight.data, self.low_ll_weight.data = \
        self._apply_noise(self.high_hh_weight.data), self._apply_noise(self.high_hl_weight.data), \
        self._apply_noise(self.low_lh_weight.data), self._apply_noise(self.low_ll_weight.data)

        self.high_hh_weight, self.high_hl_weight, self.low_lh_weight, self.low_ll_weight = \
        nn.Parameter(self.high_hh_weight), nn.Parameter(self.high_hl_weight), nn.Parameter(self.low_lh_weight), nn.Parameter(self.low_ll_weight)

        self.high_hh_bias, self.high_hl_bias, self.low_lh_bias, self.low_ll_bias = \
        nn.Parameter(self.high_hh_bias), nn.Parameter(self.high_hl_bias), nn.Parameter(self.low_lh_bias), nn.Parameter(self.low_ll_bias)
        

        self.avgpool = nn.AvgPool2d(2)
 
    def forward(self, x):
        if self.oct_type in ('first', 'A'):
            high_group, low_group = x[:, :self.num_high_in_channels, :, :], x[:, self.num_high_in_channels:, :, :]
        else:
            high_group, low_group = x

        high_group_hh = F.conv2d(high_group, self.high_hh_weight, self.high_hh_bias, self.stride,
                        self.padding, self.dilation, self.groups)
        high_group_pooled = self.avgpool(high_group)

        if self.oct_type in ('first', 'A'):
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding, self.dilation, self.groups)
            high_group_out, low_group_out = high_group_hh, high_group_hl

            return high_group_out, low_group_out

        elif self.oct_type in ('last', 'C'):
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            high_group_out = high_group_hh + low_group_upsampled

            return high_group_out

        else:
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding, self.dilation, self.groups)
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            low_group_ll = F.conv2d(low_group, self.low_ll_weight, self.low_ll_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
            high_group_out = high_group_hh + low_group_upsampled
            low_group_out = high_group_hl + low_group_ll

        return high_group_out, low_group_out

    @staticmethod
    def _apply_noise(tensor, mu=0, sigma=0.0001):
        noise = torch.normal(mean=torch.ones_like(tensor) * mu, std=torch.ones_like(tensor) * sigma)

        return tensor + noise


class OctReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu_h, self.relu_l = nn.ReLU(inplace), nn.ReLU(inplace)

    def forward(self, x):
        h, l = x

        return self.relu_h(h), self.relu_l(l)


class OctMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.maxpool_h = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool_l = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        h, l = x

        return self.maxpool_h(h), self.maxpool_l(l)


class Error(Exception):
    """Base-class for all exceptions rased by this module."""


class InvalidOctType(Error):
    """There was a problem in the OctConv type."""
