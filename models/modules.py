import torch

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode="zeros", groups=1, bias=False):
        super(Conv, self).__init__()
        assert padding_mode in ['zeros', 'circular']
        self.padding_mode = padding_mode
        if not isinstance(kernel_size, int):
            self.padding = [(i - 1) // 2 for i in kernel_size]
        else:
            self.padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups, bias=bias)
            
    def forward(self, x):
        # dim 0: angle; dim 1: delay
        # Apply circular padding on the angle domain
        padding_mode = 'circular' if self.padding_mode == 'circular' else 'constant'
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode=padding_mode)
        
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode='constant', value=0)
        x = self.conv(x)
        return x


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, bias=False, factorization=False):  
        modules = []
        if not factorization:
            modules.append(("conv", Conv(in_channels, out_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, bias=bias)))
        else:
            modules.append(("conv_1", Conv(in_channels, out_channels, [kernel_size, 1], 1, padding_mode=padding_mode, groups=groups, bias=bias)))   # double conv but only one stride decade
            modules.append(("conv_2", Conv(out_channels, out_channels, [1, kernel_size], stride, padding_mode=padding_mode, groups=groups, bias=bias)))
        
        modules.append(("bn", nn.BatchNorm2d(out_channels)))
        modules.append(("act", nn.SiLU()))
        
        super(ConvBN, self).__init__(OrderedDict(modules))


class MultiConv(nn.Module):
    ''' Convolutional module with Multi-resolution mechnism'''
    def __init__(self, in_channels, out_channels, kernel_in, kernel_1, kernel_2, padding_mode='zeros', shortcut=True, groups=1, factorization=False):
        super(MultiConv, self).__init__()
        self.in_conv = ConvBN(in_channels, out_channels, kernel_in, 1, padding_mode=padding_mode, groups=groups, factorization=factorization)
        self.conv1 = ConvBN(out_channels, out_channels, kernel_1, 1, padding_mode=padding_mode, groups=groups, factorization=factorization)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_2, 1, padding_mode=padding_mode, groups=groups, factorization=factorization)        
        self.add = shortcut
        self.conv1d = ConvBN(2 * out_channels, out_channels, 1, 1, padding_mode="zeros", groups=groups)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.in_conv(x)
        return self.act(out + self.conv1d(torch.cat((self.conv1(out), self.conv2(out)), dim=1))) if self.add else self.act(self.conv1d(torch.cat((self.conv1(out), self.conv2(out)), dim=1)))    

class UpCatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        super(UpCatConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(in_channels * 2, in_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, factorization=factorization)),
            ("conv2", ConvBN(in_channels, out_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, factorization=factorization))]
        ))
            
    def forward(self, x1, x2):
        return self.conv(torch.cat((self.up(x1), x2), dim=1))       

class DownCatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        super(DownCatConv, self).__init__()
        self.downconv = ConvBN(in_channels, in_channels, kernel_size, stride=2, padding_mode=padding_mode, groups=groups, factorization=factorization)
        self.conv = ConvBN(in_channels * 2, out_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, factorization=factorization)
    
    def forward(self, x1, x2):
        return self.conv(torch.cat((self.downconv(x1), x2), dim=1))
    
class HeadConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        layer = []
        if not factorization:
            layer.append(("conv", Conv(in_channels, out_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, bias=False)))
        else:
            layer.append(("conv_1", Conv(in_channels, out_channels, [kernel_size, 1], stride, padding_mode=padding_mode, groups=groups, bias=False)))
            layer.append(("conv_2", Conv(out_channels, out_channels, [1, kernel_size], stride, padding_mode=padding_mode, groups=groups, bias=False)))
        layer.append(("bn", nn.BatchNorm2d(out_channels)))
        layer.append(("act", nn.LeakyReLU(negative_slope=0.3, inplace=True)))
        super(HeadConv, self).__init__(OrderedDict(layer))
