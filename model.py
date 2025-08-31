# Define network components here
import os
import settings as settings
os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu_ids
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from collections import OrderedDict
from timm.models.layers import DropPath
from CLIP import clip as clip

clip_model, preprocess = clip.load("./CLIP/ViT-B-32.pt")  # ViT-B/32
clip_model.cuda()

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=False):
#         #如果 shared_b 被设置为 True，则会创建一个共享的参数 b，它将被用于计算 out_l 和 out_r。
# 如果 shared_b 被设置为 False，则会创建两个独立的参数 b_l 和 b_r，分别用于计算 out_l 和 out_r。
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

class MuGIBlock_1to2(nn.Module):
    def __init__(self, c, shared_b=False):
#         #如果 shared_b 被设置为 True，则会创建一个共享的参数 b，它将被用于计算 out_l 和 out_r。
# 如果 shared_b 被设置为 False，则会创建两个独立的参数 b_l 和 b_r，分别用于计算 out_l 和 out_r。
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp_l = x
        inp_r = x
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

## Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.PReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y

class Decoupling(nn.Module):
    def __init__(self, channel=64):
        super(Decoupling, self).__init__()
        self.conv_T1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_main1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_T2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))

        self.fuse1 = nn.Sequential(nn.Conv2d(2*channel, channel,3, 1, 1), nn.LeakyReLU(0.2))
        self.fuse2 = nn.Sequential(nn.Conv2d(2*channel, channel,3, 1, 1), nn.LeakyReLU(0.2))
        # self.dwconv1 = nn.Sequential(nn.Conv2d(channel, channel,3, 1, 1,groups=channel), nn.LeakyReLU(0.2))
        # self.dwconv2 = nn.Sequential(nn.Conv2d(channel, channel,3, 1, 1,groups=channel), nn.LeakyReLU(0.2))
        self.attn1 = SALayer(channel)
        self.attn2 = SALayer(channel)

    def forward(self, x_main,x_T,x_R):
        x_main1 = self.conv_main1(x_main)
        x_T1 = self.conv_T1(x_T)
        x_R1 = self.conv_R1(x_R)
        x1 = self.fuse1(torch.cat([x_main1,x_T1],dim=1))+x_T
        x2 = self.fuse2(torch.cat([x_main1,x_R1],dim=1))+x_R
        x1_attn = self.attn1(x1)
        x2_attn = self.attn2(x2)
        x_T_out = x1_attn + self.conv_T2(x2 - x2_attn)+x1
        x_R_out = x2_attn + self.conv_R2(x1 - x1_attn)+x2
        return x_T_out,x_R_out

class two_Decoupling(nn.Module):
    def __init__(self, channel=64):
        super(two_Decoupling, self).__init__()
        self.conv_T1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_main1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_T2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))

        self.fuse1 = nn.Sequential(nn.Conv2d(2*channel, channel,3, 1, 1), nn.LeakyReLU(0.2))
        self.fuse2 = nn.Sequential(nn.Conv2d(2*channel, channel,3, 1, 1), nn.LeakyReLU(0.2))
        # self.dwconv1 = nn.Sequential(nn.Conv2d(channel, channel,3, 1, 1,groups=channel), nn.LeakyReLU(0.2))
        # self.dwconv2 = nn.Sequential(nn.Conv2d(channel, channel,3, 1, 1,groups=channel), nn.LeakyReLU(0.2))
        self.attn1 = SALayer(channel)
        self.attn2 = SALayer(channel)

    def forward(self, x_T,x_R):
        # x_main1 = self.conv_main1(x_main)
        x1 = self.conv_T1(x_T)+x_T
        x2 = self.conv_R1(x_R)+x_R
        # x1 = self.fuse1(torch.cat([x_main1,x_T1],dim=1))+x_T
        # x2 = self.fuse2(torch.cat([x_main1,x_R1],dim=1))+x_R
        x1_attn = self.attn1(x1)
        x2_attn = self.attn2(x2)
        x_T_out = x1_attn + self.conv_T2(x2 - x2_attn)
        x_R_out = x2_attn + self.conv_R2(x1 - x1_attn)
        return x_T_out,x_R_out

class single_Decoupling(nn.Module):
    def __init__(self, channel=64):
        super(single_Decoupling, self).__init__()
        self.conv_T1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R1 = nn.Sequential(LayerNorm2d(channel),
                nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_T2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.conv_R2 = nn.Sequential(nn.Conv2d(channel, channel, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel))
        self.attn1 = SALayer(channel)
        self.attn2 = SALayer(channel)

    def forward(self, x):
        x_T1 = self.conv_T1(x)
        x_R1 = self.conv_R1(x)
        x_T1_attn = self.attn1(x_T1)+x
        x_R1_attn = self.attn2(x_R1)+x
        x_T_out = x_T1_attn + self.conv_T2(x_R1 - x_R1_attn)
        x_R_out = x_R1_attn + self.conv_R2(x_T1 - x_T1_attn)
        return x_T_out,x_R_out


class FeaturePyramidVGG(nn.Module):
    def __init__(self, out_channels=64, shared_b=False):
        super().__init__()
        self.device = 'cuda'
        self.block5 = DualStreamSeq(
            MuGIBlock(512, shared_b),
            DualStreamBlock(nn.UpsamplingBilinear2d(scale_factor=2.0)),
        )

        self.block4 = DualStreamSeq(
            MuGIBlock(512, shared_b)
        )

        self.ch_map4 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(256, shared_b)
        )

        self.block3 = DualStreamSeq(
            MuGIBlock(256, shared_b)
        )

        self.ch_map3 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(128, shared_b)
        )

        self.block2 = DualStreamSeq(
            MuGIBlock(128, shared_b)
        )

        self.ch_map2 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(64, shared_b)
        )

        self.block1 = DualStreamSeq(
            MuGIBlock(64, shared_b),
        )

        self.ch_map1 = DualStreamSeq(
            DualStreamBlock(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)),
            MuGIBlock(128, shared_b),
            DualStreamBlock(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)),
            MuGIBlock(32, shared_b),
        )

        self.block_intro = DualStreamSeq(
            DualStreamBlock(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)),
            MuGIBlock(32, shared_b)
        )

        self.ch_map0 = DualStreamBlock(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, inp, vgg_feats):
        # 64,128,256,512,512
        vf1, vf2, vf3, vf4, vf5 = vgg_feats
        # 512=>512,512=>512
        f5_l, f5_r = self.block5(vf5)
        f4_l, f4_r = self.block4(vf4)
        f4_l, f4_r = self.ch_map4(torch.cat([f5_l, f4_l], dim=1), torch.cat([f5_r, f4_r], dim=1))
        # 256 => 256
        f3_l, f3_r = self.block3(vf3)
        # (256+256,256+256)->(128,128)
        f3_l, f3_r = self.ch_map3(torch.cat([f4_l, f3_l], dim=1), torch.cat([f4_r, f3_r], dim=1))
        # (128+128,128+128)->(64,64)
        f2_l, f2_r = self.block2(vf2)
        f2_l, f2_r = self.ch_map2(torch.cat([f3_l, f2_l], dim=1), torch.cat([f3_r, f2_r], dim=1))
        # (64+64,64+64)->(32,32)
        f1_l, f1_r = self.block1(vf1)
        f1_l, f1_r = self.ch_map1(torch.cat([f2_l, f1_l], dim=1), torch.cat([f2_r, f1_r], dim=1))
        # 64
        f0_l, f0_r = self.block_intro(inp, inp)
        f0_l, f0_r = self.ch_map0(torch.cat([f1_l, f0_l], dim=1), torch.cat([f1_r, f0_r], dim=1))
        return f0_l, f0_r

class R2Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SinBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2),
            SimpleGate(),
            CABlock(c),
            nn.Conv2d(c, c, 1)
        )

        self.block2 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1)
        )

        self.a = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.block1(inp)
        x_skip = inp + x * self.a
        x = self.block2(x_skip)
        out = x_skip + x * self.b
        return out

class LRM(nn.Module):
    def __init__(self, in_channels=48, num_blocks=[2, 4]):
        super().__init__()
        self.device = 'cuda'
        channel = in_channels * 2
        self.intro = DualStreamBlock(nn.Conv2d(in_channels, channel, 1))
        self.blocks_inter = DualStreamSeq(*[R2Block(channel) for _ in range(num_blocks[0])])
        self.blocks_merge = nn.Sequential(*[SinBlock(channel * 2) for _ in range(num_blocks[1])])
        self.tail = nn.Sequential(
            nn.Conv2d(channel * 2, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, ft, fr):
        ft, fr = self.intro(ft, fr)
        ft, fr = self.blocks_inter(ft, fr)
        fs = self.blocks_merge(torch.cat([ft, fr], dim=1))
        out = self.tail(fs)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Language_guided_Spatial_Channel_Cross_Transformer(nn.Module):
    def __init__(self, dim=64,num_heads=1,mlp_ratio=4,bias=True,attn_drop=0,proj_drop=0,drop_path=0):
        super(Language_guided_Spatial_Channel_Cross_Transformer, self).__init__()
        self.num_heads = num_heads
        self.fc1 = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, dim , bias=bias))
        self.fc2 = nn.Linear(dim, dim , bias=bias)
        self.fc3 = nn.Linear(1, 1 , bias=bias)
        self.fc_text = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, dim , bias=bias))
        self.logit1_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.logit2_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x,text): #text:B,1,C
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2) #B,L,C
        B, L, C = x.shape
        shortcut = x

        # Language-guided Spatial-Channel Cross Attention (LSCA)
        F_I = self.fc1(x)
        F_S = F_I.mean(1,keepdim=True) #B,1,C
        F_C = F_I.mean(2,keepdim=True) #B,L,1

        F_S = self.fc2(F_S) #B,1,C
        F_C = self.fc3(F_C) #B,L,1
        if text != None:
            F_L = self.fc_text(text) #B,1,C
        else:
            F_L = F_S #B,1,C

        # cosine attention
        M_SL = (F.normalize(F_S, dim=-2).transpose(-2, -1) @ F.normalize(F_L, dim=-2)) #B,C,C
        logit1_scale = torch.clamp(self.logit1_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        M_SL = M_SL * logit1_scale
        M_SL = M_SL.softmax(dim=-1)
        M_SL = self.attn_drop1(M_SL) #B,C,C

        # cosine attention
        M_LC = (F.normalize(F_C, dim=-1)@ F.normalize(F_L, dim=-2)) #B,L,C
        logit2_scale = torch.clamp(self.logit2_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        M_LC = M_LC * logit2_scale
        M_LC = M_LC.softmax(dim=-1)
        M_LC = self.attn_drop2(M_LC) #B,L,C

        M_LCSL = (M_LC @ M_SL).reshape(B, L, C) #B,L,C

        x = self.proj(M_LCSL)
        x = self.proj_drop(x)

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        out = x.transpose(1,2).view(B, C ,H, W)
        return out

class competition_avgallocation(nn.Module):
    def __init__(self, inchannel=512,outchannel=64,ratio=8):
        super(competition_avgallocation, self).__init__()
        self.fc1_max = nn.Sequential(nn.Linear(2*inchannel, 2*inchannel//ratio), nn.LeakyReLU(0.2))
        self.fc2_max = nn.Sequential(nn.Linear(inchannel//ratio, inchannel//ratio), nn.LeakyReLU(0.2))
        self.fc1_avg = nn.Sequential(nn.Linear(2*inchannel, 2*inchannel//ratio), nn.LeakyReLU(0.2))
        self.fc2_avg = nn.Sequential(nn.Linear(inchannel//ratio, inchannel//ratio), nn.LeakyReLU(0.2))
        self.fuse = nn.Sequential(nn.Linear(2*inchannel//ratio, outchannel), nn.LeakyReLU(0.2))

    def forward(self, x1,x2):
        B, L, C = x1.size()
        gap = torch.cat([x1,x2],dim=2)
        gc_max = self.fc1_max(gap).reshape([B, 2, -1])
        # 个人注释，下面这行是竞争
        gc_max, _ = gc_max.max(dim=1)
        # 个人注释，上面这行是竞争
        gc_max = self.fc2_max(gc_max)
        gc_avg = self.fc1_avg(gap).reshape([B, 2, -1])
        # 个人注释，下面这行是竞争
        gc_avg = gc_avg.mean(dim=1)
        # 个人注释，上面这行是竞争
        gc_avg = self.fc2_avg(gc_avg)
        out = self.fuse(torch.cat([gc_max,gc_avg],dim=1))

        return out

class Adaptive_Language_Calibration_Module(nn.Module):
    def __init__(self, channel):
        super(Adaptive_Language_Calibration_Module, self).__init__()
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)
        self.fc1_img = nn.Sequential(nn.Linear(channel, channel),nn.LeakyReLU(0.2))
        self.fc2_img = nn.Sequential(nn.Linear(channel, channel),nn.LeakyReLU(0.2))
        self.fc1_text = nn.Sequential(nn.Linear(channel, channel),nn.LeakyReLU(0.2))
        self.fc2_text = nn.Sequential(nn.Linear(channel, channel),nn.LeakyReLU(0.2))
        self.fc_out = nn.Sequential(nn.Linear(channel, channel),nn.LeakyReLU(0.2))
        self.fuse = nn.Sequential(nn.Linear(2*channel, channel),nn.LeakyReLU(0.2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img,text): #text (B,1,C)
        img_pool = self.pool(img).flatten(2).transpose(1,2)
        img_spapool = self.norm1(img_pool) # B,1,C
        img_feature = self.fc1_img(img_spapool) # B,1,C
        text_feature = self.fc1_text(self.norm2(text)) # B,1,C
        x_fuse = self.fuse(torch.cat([img_feature, text_feature],dim=-1))  # B,1,C
        sig = self.sigmoid(x_fuse)
        out = self.fc2_img(img_feature) * (1-sig) + sig * self.fc2_text(text_feature)
        out = self.fc_out(out) + img_pool + text
        return out


class Language_aware_Competition_Attention_Module(nn.Module):
    def __init__(self, channel, reduction=4, bias=True):
        super(Language_aware_Competition_Attention_Module, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)
        self.fc1 = nn.Sequential(nn.Linear(channel, channel,bias=bias))
        self.attn_channel = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=bias),
                nn.Linear(channel // reduction, channel, bias=bias),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x,text): # x (B,C,H,W)
        x_pool = self.norm1(self.pool(x).flatten(2).transpose(1,2)) # B,1,C
        x_attn_channel = self.attn_channel(x_pool) # B,1,C

        if text != None:
            text = self.fc1(self.norm2(text)).transpose(1,2) # B,C,1
            similarity_matrix = text @ x_pool # B,C,C
            similarity_cores = torch.mean(similarity_matrix,dim=1,keepdim=True) # B,1,C
            adjustment_vector = self.sig(similarity_cores)
            y = adjustment_vector * text.transpose(1,2) + (1 - adjustment_vector) * x_attn_channel # B,1,C
        else:
            y = x_attn_channel #B,1,C

        out = y.transpose(1,2).unsqueeze(-1) * x + x

        return out

class  Multi_receptive_Field_Decoupling_Module(nn.Module):
    def __init__(self, channel):
        super(Multi_receptive_Field_Decoupling_Module, self).__init__()
        # 1x1, 3x3, 5x5 and 7x7 group convolution layers
        self.channeldiv = channel // 4
        self.conv1x1 = nn.Sequential(nn.Conv2d(self.channeldiv, self.channeldiv, kernel_size=1, groups=self.channeldiv),nn.LeakyReLU(0.2))
        self.conv3x3 = nn.Sequential(nn.Conv2d(self.channeldiv, self.channeldiv, kernel_size=3, padding=1, groups=self.channeldiv),nn.LeakyReLU(0.2))
        self.conv5x5 = nn.Sequential(nn.Conv2d(self.channeldiv, self.channeldiv, kernel_size=5, padding=2, groups=self.channeldiv),nn.LeakyReLU(0.2))
        self.conv7x7 = nn.Sequential(nn.Conv2d(self.channeldiv, self.channeldiv, kernel_size=7, padding=3, groups=self.channeldiv),nn.LeakyReLU(0.2))
        self.fuse = nn.Sequential(nn.Conv2d(channel//2, channel//2, kernel_size=1),nn.LeakyReLU(0.2))

    def forward(self, x,y):
        # Split input along the channel dimension
        x1,x2,x3,x4 = torch.chunk(x, 4, dim=1)
        y1,y2,y3,y4 = torch.chunk(y, 4, dim=1)
        # Apply 1x1, 3x3, 5x5 and 7x7  convolutions to each split

        x_out1x1 = self.conv1x1(x1)
        x_out1x1_t,x_out1x1_r = torch.chunk(x_out1x1, 2, dim=1)

        y_out1x1 = self.conv1x1(y1)
        y_out1x1_t,y_out1x1_r = torch.chunk(y_out1x1, 2, dim=1)

        x_out3x3 = self.conv3x3(x2)
        x_out3x3_t,x_out3x3_r = torch.chunk(x_out3x3, 2, dim=1)

        y_out3x3 = self.conv3x3(y2)
        y_out3x3_t,y_out3x3_r = torch.chunk(y_out3x3, 2, dim=1)

        x_out5x5 = self.conv5x5(x3)
        x_out5x5_t,x_out5x5_r = torch.chunk(x_out5x5, 2, dim=1)

        y_out5x5 = self.conv5x5(y3)
        y_out5x5_t,y_out5x5_r = torch.chunk(y_out5x5, 2, dim=1)

        x_out7x7 = self.conv7x7(x4)
        x_out7x7_t,x_out7x7_r = torch.chunk(x_out7x7, 2, dim=1)

        y_out7x7 = self.conv7x7(y4)
        y_out7x7_t,y_out7x7_r = torch.chunk(y_out7x7, 2, dim=1)

        # Concatenate the outputs along the channel dimension
        out_x = self.fuse(torch.cat([x_out1x1_t*y_out1x1_t,x_out3x3_t*y_out3x3_t,x_out5x5_t*y_out5x5_t,x_out7x7_t*y_out7x7_t], dim=1))
        out_y = self.fuse(torch.cat([x_out1x1_r*y_out1x1_r,x_out3x3_r*y_out3x3_r,x_out5x5_r*y_out5x5_r,x_out7x7_r*y_out7x7_r], dim=1))

        return out_x,out_y

class Language_aware_Separation_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, 2*c , 1),
            ),
            Multi_receptive_Field_Decoupling_Module(2*c),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)),
            Multi_receptive_Field_Decoupling_Module(2*c))

        self.LCAM = Language_aware_Competition_Attention_Module(c)

        self.block3 = DualStreamBlock(nn.Conv2d(c, c, 1))
        self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, F_I_T, F_I_R, text_T, text_R):
        x, y = self.block1(F_I_T, F_I_R)
        x_skip, y_skip = F_I_T + x * self.a_l, F_I_R + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        x, y = self.LCAM(x,text_T), self.LCAM(y,text_R)
        x, y = self.block3(x, y)
        out_T, out_R = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_T, out_R

# Adaptive Language-aware Network
class ALANet(nn.Module):
    def __init__(self, channels=[64,128,128,160,160], enc_blk_nums=[2, 2, 2, 2],
                 dec_blk_nums=[2, 2, 2, 2], middle_blk_num=4):
        super().__init__()
        c0 = channels[0]
        c1 = channels[1]
        c2 = channels[2]
        c3 = channels[3]
        c4 = channels[4]
        self.convert_T = nn.Sequential(nn.Conv2d(3, c0, 3,1,1), nn.LeakyReLU(0.2))
        self.convert_R = nn.Sequential(nn.Conv2d(3, c0, 3,1,1), nn.LeakyReLU(0.2))
        self.ending = nn.Sequential(nn.Conv2d(c0, 3, 3,1,1), nn.LeakyReLU(0.2))

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.down = nn.AvgPool2d(2,2)
        self.fuse_enc = nn.ModuleList()
        self.fuse_dec = nn.ModuleList()

        for i,num in enumerate(enc_blk_nums):
            c = channels[i] #i [0,1,2,3]
            c_pre = c
            if i != 0:
                c_pre = channels[i - 1] #c_pre [0,0,1,2]
            self.fuse_enc.append(DualStreamBlock(
                    *[nn.Conv2d(c+c_pre,c, 3, 1, 1), nn.LeakyReLU(0.2)]
            ))

            self.encoders.append(
                nn.ModuleList(
                    [Language_aware_Separation_Block(c) for _ in range(num)]
                )
            )

        self.fuse_mid = DualStreamBlock(*[nn.Conv2d(c3+c4 , c4, 3, 1, 1), nn.LeakyReLU(0.2)])
        self.ALCM = Adaptive_Language_Calibration_Module(c4)
        self.LSCT = Language_guided_Spatial_Channel_Cross_Transformer(c4)
        self.middle_blks = nn.ModuleList(
            [Language_aware_Separation_Block(c4) for _ in range(middle_blk_num)]
        )

        for i, num in enumerate(dec_blk_nums):
            c = channels[len(dec_blk_nums) - 1 - i] #c [3,2,1,0]
            c_pre = channels[len(dec_blk_nums) - i] #c_pre [4,3,2,1]
            self.fuse_dec.append(DualStreamBlock(
                *[nn.Conv2d(c + c_pre, c, 3, 1, 1), nn.LeakyReLU(0.2)]
            ))

            self.decoders.append(
                nn.ModuleList(
                    [Language_aware_Separation_Block(c) for _ in range(num)]
                )
            )

        self.fc_level0 = nn.Sequential(nn.Linear(512, c0), nn.LeakyReLU(0.2))
        self.fc_level1 = nn.Sequential(nn.Linear(512, c1), nn.LeakyReLU(0.2))
        self.fc_level2 = nn.Sequential(nn.Linear(512, c2), nn.LeakyReLU(0.2))
        self.fc_level3 = nn.Sequential(nn.Linear(512, c3), nn.LeakyReLU(0.2))
        self.fc_level4 = nn.Sequential(nn.Linear(512, c4), nn.LeakyReLU(0.2))

        self.vgg_conv0 = nn.Sequential(nn.Conv2d(64,c0,1,1),nn.LeakyReLU(0.2))
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(128,c1,1,1),nn.LeakyReLU(0.2))
        self.vgg_conv2 = nn.Sequential(nn.Conv2d(256,c2,1,1),nn.LeakyReLU(0.2))
        self.vgg_conv3 = nn.Sequential(nn.Conv2d(512,c3,1,1),nn.LeakyReLU(0.2))
        self.vgg_conv4 = nn.Sequential(nn.Conv2d(512,c4,1,1),nn.LeakyReLU(0.2))
        self.vgg_LASB_level0 = Language_aware_Separation_Block(c0)
        self.vgg_LASB_level1 = Language_aware_Separation_Block(c1)
        self.vgg_LASB_level2 = Language_aware_Separation_Block(c2)
        self.vgg_LASB_level3 = Language_aware_Separation_Block(c3)
        self.vgg_LASB_level4 = Language_aware_Separation_Block(c4)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 32
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def forward(self, input,caption_T,caption_R,feature_vgg):
        input,ori_size = self.check_image_size(input)
        B, C, H, W = input.shape
        x = self.convert_T(input)
        y = self.convert_R(input)

        encs = []

        # Language feature initialization
        text_T_level0, text_T_level1, text_T_level2, text_T_level3, text_T_level4, \
        text_R_level0, text_R_level1, text_R_level2, text_R_level3, text_R_level4 = [None] * 10

        # Language Feature Extraction Branch
        if caption_T != None:
            tokenize_T = clip.tokenize(caption_T).cuda()
            text_features_T = clip_model.encode_text(tokenize_T)
            text_T = text_features_T.view(B, 1, -1).type(torch.float32)
            text_T_level0 = self.fc_level0(text_T)
            text_T_level1 = self.fc_level1(text_T)
            text_T_level2 = self.fc_level2(text_T)
            text_T_level3 = self.fc_level3(text_T)
            text_T_level4 = self.fc_level4(text_T)
        if caption_R != None:
            tokenize_R = clip.tokenize(caption_R).cuda()
            text_features_R = clip_model.encode_text(tokenize_R)
            text_R = text_features_R.view(B, 1, -1).type(torch.float32)
            text_R_level0 = self.fc_level0(text_R)
            text_R_level1 = self.fc_level1(text_R)
            text_R_level2 = self.fc_level2(text_R)
            text_R_level3 = self.fc_level3(text_R)
            text_R_level4 = self.fc_level4(text_R)

        # Perception Decoupling Branch
        vgg_level0, vgg_level1, vgg_level2, vgg_level3, vgg_level4 = feature_vgg #[H,H/2,H/4,H/8,H/16] [64,128,256,512,512]
        vgg_level0 = self.vgg_conv0(vgg_level0)
        vgg_level1 = self.vgg_conv1(vgg_level1)
        vgg_level2 = self.vgg_conv2(vgg_level2)
        vgg_level3 = self.vgg_conv3(vgg_level3)
        vgg_level4 = self.vgg_conv4(vgg_level4)
        
        vgg_T_level0,vgg_R_level0 = self.vgg_LASB_level0(vgg_level0,vgg_level0,text_T_level0,text_R_level0)
        vgg_T_level1,vgg_R_level1 = self.vgg_LASB_level1(vgg_level1,vgg_level1,text_T_level1,text_R_level1)
        vgg_T_level2,vgg_R_level2 = self.vgg_LASB_level2(vgg_level2,vgg_level2,text_T_level2,text_R_level2)
        vgg_T_level3,vgg_R_level3 = self.vgg_LASB_level3(vgg_level3,vgg_level3,text_T_level3,text_R_level3)
        vgg_T_level4,vgg_R_level4 = self.vgg_LASB_level4(vgg_level4,vgg_level4,text_T_level4,text_R_level4)  
        vgg_T = vgg_T_level0
        vgg_R = vgg_R_level0


        # Language-aware Separation Branch
        for i, (fuse_enc, encoder) in enumerate(zip(self.fuse_enc, self.encoders)): #i,[0,1,2,3]
            # Construct variable names
            vgg_T_name = f"vgg_T_level{i}"
            vgg_R_name = f"vgg_R_level{i}"

            # Get the value of the variable
            vgg_T = locals()[vgg_T_name]
            vgg_R = locals()[vgg_R_name]

            text_T_name = f"text_T_level{i}"
            text_T = locals()[text_T_name]
            text_R_name = f"text_R_level{i}"
            text_R = locals()[text_R_name]

            x, y = fuse_enc(torch.cat([x, vgg_T], dim=1), torch.cat([y, vgg_R], dim=1))
            for j, module in enumerate(encoder):
                x, y = module(x, y,text_T,text_R)
            encs.append((x, y))
            x, y = self.down(x), self.down(y)

        x, y = self.fuse_mid(torch.cat([x, vgg_T_level4], dim=1), torch.cat([y, vgg_R_level4], dim=1))
        if caption_T != None:
            text_T_level4 =self.ALCM(x,text_T_level4)
        if caption_R != None:
            text_R_level4 =self.ALCM(y,text_R_level4)
        x, y = self.LSCT(x,text_T_level4), self.LSCT(y,text_R_level4)
        for j, module in enumerate(self.middle_blks):
            x, y = module(x, y, text_T_level4, text_R_level4)

        for i,(fuse_dec, decoder, (enc_x_skip, enc_y_skip)) in enumerate(zip(self.fuse_dec,self.decoders, encs[::-1])):
            x, y = self.up(x),self.up(y)
            x, y = fuse_dec(torch.cat([x , enc_x_skip],dim=1),torch.cat([y , enc_y_skip],dim=1))

            text_T_name = f"text_T_level{3-i}"
            text_T = locals()[text_T_name]
            text_R_name = f"text_R_level{3-i}"
            text_R = locals()[text_R_name]

            for j, module in enumerate(decoder):
                x, y = module(x, y, text_T, text_R)

        x, y = self.ending(x), self.ending(y)
        x_out, y_out = self.restore_image_size(x,ori_size), self.restore_image_size(y,ori_size)
        return x_out, y_out



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=False)  # .to(device)  # .cuda()
        self.vgg.load_state_dict(torch.load('./models/vgg19-dcbb9e9d.pth'))
        print('Vgg loaded successfully!')
        self.vgg.eval()
        self.vgg = self.vgg.features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out

