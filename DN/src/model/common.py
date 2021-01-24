import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.parameter import Parameter

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True




class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class MABlock(nn.Module):
    def __init__(self, conv, kernel_size=3, bias=True, act=nn.ReLU(True)):
        super(MABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k1 = Parameter(torch.zeros(1))
        #self.contrast = stdv_channels
        self.softmax = nn.Softmax(dim=1)
        m =[]
        m.append(conv(1, 1, kernel_size, bias=bias))
        m.append(act)
        m.append(conv(1, 1, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)
        self.reset_parameters()

    def reset_parameters(self):
        last_zero_init(self.body)

    def NCA(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        Apx = self.avg_pool(input_x)
        #Vpx = self.contrast(input_x)
        px = Apx
        px = self.softmax(px)    # N*c*1*1
        px = px.view(batch, 1, channel, 1)    # N*1*c*1
        input_x = input_x.view(batch, height * width, channel)   # N*HW*c
        input_x = input_x.unsqueeze(1)    # N*1*HW*c
        context = torch.matmul(input_x, px)     # N*1*HW*1
        context = context.view(batch, 1, height, width)    # N*1*H*W
        return context
    def forward(self, x):
        context = self.NCA(x)
        context = self.body(context)
        #x = x + context
        return self.k1*context
class ContextBlock2d(nn.Module):

    def __init__(self, inplanes):
        super(ContextBlock2d, self).__init__()
        self.inplanes = inplanes
        self.planes = inplanes//16
        self.k2 = Parameter(torch.zeros(1))
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        #out = out + channel_add_term

        return self.k2*channel_add_term

class AFF(nn.Module):
    def __init__(self, inchannels, conv, kernel_size=3, bias=True, act=nn.ReLU(True)):
        super(AFF, self).__init__()
        
        self.k3 = Parameter(torch.zeros(1))
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        

    def NBA(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x #N*C*H*W*K
        #input_y = y #N*1*1*K*1
        y = self.softmax(y)
        context = torch.matmul(input_x, y) #N*C*H*W*1
        context = context.view(batch, channel, height, width) #N*C*H*W

        return context
    def forward(self, x, y):
        out1 = self.NBA(x,y)
        return self.k3*out1

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

