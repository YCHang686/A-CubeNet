from model import common

import torch.nn as nn

import torch

def make_model(args, parent=False):
    return ACubeNet(args)
class MCPN(nn.Module):
    def __init__(self, channel,conv=common.default_conv):
        super(MCPN, self).__init__()
        self.MA = common.MABlock(conv)
        self.context = common.ContextBlock2d(channel)
    def forward(self, x):
        x1 = self.MA(x)
        x2 = self.context(x)
        x = x + x1 + x2
        return x


## Residual Dual Attention Unit (RDAU)
class RDAU(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RDAU, self).__init__()
        modules_body = []
        modules_body.append(common.ResBlock(conv, n_feat, kernel_size=3, act=act, res_scale=1))
        modules_body.append(MCPN(n_feat))
        self.body = nn.Sequential(*modules_body)
        

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        #res += x
        return res

## Residual Dual Attention Group (RDAG)
class RDAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(RDAG, self).__init__()
        modules_body = []
        modules_body = [
            RDAU(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Attention Cube Network (ACubeNet)
class ACubeNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ACubeNet, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.aff = common.AFF(n_feats, conv)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        self.body = nn.ModuleList()
        for _ in range(n_resgroups):
            self.body.append(RDAG(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))

        self.sq = nn.ModuleList()
        for _ in range(n_resgroups):
            self.sq.append(conv(n_feats, 1, kernel_size=1))

        m_VFF = []
        m_VFF.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.VFF = nn.Sequential(*m_VFF)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        y = x
        RBs_sq = []
        RBs_up = []
        for i in range(4):
            y = self.body[i](y) #N*C*H*W
            z = self.avg_pool(y) #N*C*1*1
            z = self.sq[i](z) #N*1*1*1
            z = z.unsqueeze(-2) #N*1*1*1*1
            t = y.unsqueeze(-1) #N*C*H*W*1
            #RBs_out.append(y) #K_N*C*H*W
            RBs_up.append(t) #K_N*C*H*W*1
            RBs_sq.append(z) #K_N*1*1*1*1

        output1 = torch.cat(RBs_up,dim=-1) #N*C*H*W*K
        output2 = torch.cat(RBs_sq,dim=-2) #N*1*1*K*1
        out = self.aff(output1, output2) #N*C*H*W
        
        y = y + out
        res = self.VFF(y)

        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
