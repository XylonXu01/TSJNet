import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import cv2
from natten import NeighborhoodAttention2D as NeighborhoodAttention
import time

class rSoftMax(nn.Module):
    def __init__(self, cardinal=1, radix=2):
        super().__init__()
        self.groups = cardinal
        self.radix = radix

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(batch_size, -1, 1, 1)
        return x

class SplitAttentionBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1, radix=2, cardinal=1, reduction_factor=4, **kwargs):
        super(SplitAttentionBlock, self).__init__()
        self.radix = radix
        self.cardinal = cardinal
        self.radix_conv = nn.Sequential(Conv2d(in_channels, channels*radix, (kernel_size, kernel_size), (stride, stride),
                                               groups=radix*cardinal, **kwargs), BatchNorm2d(channels*radix),
                                        ReLU(inplace=True))
        inter_channels = max(32, int(in_channels*radix/reduction_factor))
        self.fc1 = nn.Sequential(Conv2d(channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), groups=cardinal, bias=False),
                                 BatchNorm2d(inter_channels),
                                 ReLU(inplace=True))
        self.fc2 = nn.Sequential(Conv2d(inter_channels, channels*radix, kernel_size=(1, 1), stride=(1, 1), groups=cardinal,
                                        bias=False), BatchNorm2d(channels*radix), ReLU(inplace=True))
        self.rsoftmax = rSoftMax(cardinal, radix)

    def forward(self, x):
        x = self.radix_conv(x)
        batch_size, r_channels = x.shape[:2]
        splits = torch.split(x, int(r_channels/self.radix), dim=1)
        gap = sum(splits)
        gap = F.adaptive_avg_pool2d(gap, 1)
        att = self.fc1(gap)
        att = self.fc2(att)
        att = self.rsoftmax(att)
        atts = torch.split(att, int(r_channels/self.radix), dim=1)
        output = sum([split*attention for split, attention in zip(splits, atts)])
        return output.contiguous()

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, radix=2, cardinal=1, bottleneck_width=64):
        super(BottleNeck, self).__init__()
        group_width = int(planes*(bottleneck_width/64))*cardinal
        self.radix = radix
        self.conv1 = nn.Sequential(Conv2d(in_planes, group_width, kernel_size=(1, 1), stride=(1, 1)),
                                   BatchNorm2d(group_width), ReLU(inplace=True))
        self.conv2 = SplitAttentionBlock(group_width, group_width, kernel_size=3, stride=stride, padding=1,
                                         cardinal=cardinal, radix=radix)
        self.conv3 = nn.Sequential(Conv2d(group_width, planes*self.expansion, kernel_size=(1, 1), stride=(1, 1)),
                                   BatchNorm2d(planes*self.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            self.shortcut = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                                          Conv2d(in_planes, planes*self.expansion, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                          BatchNorm2d(planes*self.expansion))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.shortcut:
            residual = self.shortcut(residual)
        output = residual+out
        return self.relu(output)

class ResNest(nn.Module):
    def __init__(self, block, num_blocks, radix=2, cardinal=1, bottleneck_width=64,inchannel=3, num_classes=176):
        super(ResNest, self).__init__()
        self.radix = radix
        self.cardinal = cardinal
        self.bottleneck_width = bottleneck_width
        self.inchannel=inchannel
        #self.deep_stem = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1), #不改变图像大小
        self.deep_stem = nn.Sequential(Conv2d(inchannel, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),

                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(128), ReLU(inplace=True))
        self.in_channels = 128
        self.layer1 = self._make_stage(block, num_blocks[0], 1, 128)
        #self.layer2 = self._make_stage(block, num_blocks[1], 2, 128)#为了不改变图像大小
        # self.layer2 = self._make_stage(block, num_blocks[1], 1, 128)
        #self.layer3 = self._make_stage(block, num_blocks[2], 2, 256)
        #self.layer4 = self._make_stage(block, num_blocks[3], 2, 512)

        # self.relu = ReLU(inplace=True)
        # self.avg = nn.AdaptiveAvgPool2d(1)

        #self.classifier = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.deep_stem(x)
        #print(x.shape)
        x = self.layer1(x)
        
        #print(x.shape)
        # x = self.layer2(x)

        #x = self.layer3(x)
        #x = self.layer4(x)
        # x = self.avg(x)
        # x = x.view(x.size(0),-1)
       #x = self.classifier(x)
        return x

    def _make_stage(self, block, num_block, stride, channels):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def ResNest(self):
        return ResNest(BottleNeck, [2, 2])
    
class ResNest2(nn.Module):
    def __init__(self, block, num_blocks, radix=2, cardinal=1, bottleneck_width=64,inchannel=3, num_classes=176):
        super(ResNest2, self).__init__()
        self.radix = radix
        self.cardinal = cardinal
        self.bottleneck_width = bottleneck_width
        self.inchannel=inchannel
        #self.deep_stem = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1), #不改变图像大小
        self.deep_stem = nn.Sequential(Conv2d(inchannel, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),

                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(128), ReLU(inplace=True))
        self.in_channels = 128
        self.layer1 = self._make_stage(block, num_blocks[0], 1, 128)
        #self.layer2 = self._make_stage(block, num_blocks[1], 2, 128)#为了不改变图像大小
        # self.layer2 = self._make_stage(block, num_blocks[1], 1, 128)
        #self.layer3 = self._make_stage(block, num_blocks[2], 2, 256)
        #self.layer4 = self._make_stage(block, num_blocks[3], 2, 512)

        # self.relu = ReLU(inplace=True)
        # self.avg = nn.AdaptiveAvgPool2d(1)

        #self.classifier = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.deep_stem(x)
        x = self.layer1(x)
        return x

    def _make_stage(self, block, num_block, stride, channels):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def ResNest(self):
        return ResNest(BottleNeck, [2, 2])
#--------------------------------------------------------------------------------------------------------------------------------------------

class MatMul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a @ b
        return out

# class AttentionBase(nn.Module):

#     def __init__(self,
#                  dim,
#                  num_heads=8,
#                  qkv_bias=False,
#                  attn_drop=0.0,
#                  proj_drop=0.0,
#                  res_kernel_size=9,
#                  sparse_reg=False,
#                 # mask_ratio=0.3
#                  ):
#         super(AttentionBase, self).__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         #self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.sparse_reg = sparse_reg
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)


#         self.dconv = nn.Conv2d(
#             in_channels=self.num_heads,
#             out_channels=self.num_heads,
#             kernel_size=(res_kernel_size, 1),
#             padding=(res_kernel_size // 2, 0),
#             bias=False,
#             groups=self.num_heads,)

#         self.kq_matmul = MatMul()
#         self.kqv_matmul = MatMul()
#         if self.sparse_reg:
#             self.qk_matmul = MatMul()
#             self.sv_matmul = MatMul()

#         self.dconv = nn.Conv2d(
#          in_channels=self.num_heads,
#          out_channels=self.num_heads,
#          kernel_size=(res_kernel_size, 1),
#          padding=(res_kernel_size // 2, 0),
#          bias=False,
#          groups=self.num_heads,
#     )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.reshape(b, c, -1).permute(0, 2, 1)
#         # print(x.shape)

#         N, L, C = x.shape
#         qkv = (
#             self.qkv(x)
#             .reshape(N, L, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

#         if self.sparse_reg:
#             attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
#             attn = attn.softmax(dim=-1)
#             mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
#             sparse = mask * attn

#         q = q / q.norm(dim=-1, keepdim=True)
#         k = k / k.norm(dim=-1, keepdim=True)
#         dconv_v = self.dconv(v)

#         attn = self.kq_matmul(k.transpose(-2, -1), v)

#         if self.sparse_reg:
#             x = (
#                 self.sv_matmul(sparse, v)
#                 + 0.5 * v
#                 + 1.0 / math.pi * self.kqv_matmul(q, attn)
#             )
#         else:
#             x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
#         x = x / x.norm(dim=-1, keepdim=True)
#         x += dconv_v
#         x = x.transpose(1, 2).reshape(N, L, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         # x = self.proj_drop(x)
#         x = x.permute(0, 2, 1).reshape(b, c, h, w)
#         # print(x.shape)
#         return x
#----------------------------------------------------------------------------------------------------------------------NAT
model_urls = {
    "nat_mini_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth",
}


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=64, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x) 
        if self.downsample is None:
            return x
        return self.downsample(x)


class NAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=64,
        kernel_size=7,
        dilations=None,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        #self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.num_features =64
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.Conv2d(self.num_features, 64, kernel_size=1)
        self.head = nn.Identity()  # 
        self.conv_head = nn.Conv2d(self.num_features, 64, kernel_size=1)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.ReLU6(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU6(inplace=True),)
        self.transposed_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4, padding=0, bias=False)
        self.transposed_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4, padding=0, bias=False)
        self.transposed_conv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = (
        #     nn.Linear(self.num_features, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.avgpool.weight = nn.Parameter(
            self.avgpool.weight[:, :self.num_features, :, :]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        # x = self.norm(x).flatten(1, 2)
        # x = self.avgpool(x.transpose(1, 2))
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        #x = self.conv_head(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = self.transposed_conv1(x2)
        x3 = self.transposed_conv2(x2)
        x4 = self.transposed_conv3(x3)
        # 添加激活函数
        x5 = nn.ReLU6(inplace=True)(x4) # inplace=True表示直接对x4进行修改
        return x5


def nat_mini(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_mini_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
#---------------------------------------------------------------------------------------------------------------------
class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        # self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.attn = nat_mini()


       # self.norm2 = LayerNorm(dim, 'WithBias')
        #self.mlp = Mlp(in_features=dim,
         #              ffn_expansion_factor=ffn_expansion_factor, )
        #self.xxxtransformer = LinAngularAttention(in_channels=256, num_heads=8, qkv_bias=False, sparse_reg=False)

    def forward(self, x):
        x1 = self.norm1(x) # 输入维度为[batch_size, 256, 64, 64],输出维度为[batch_size, 256, 64, 64]

        x1= self.attn(x1)

        x = x + x1

       #x = x + self.attn(self.norm1(x))
       # x = x + self.mlp(self.norm2(x))
        return x
#--------------------------------------------------------------------------------------------------------------------------------------------
def harris(img):
    img = img.float()
    att_0 = torch.tensor([])
    for a in range(img.shape[0]):
        att_1 = torch.tensor([])
        for b in range(img.shape[1]):
            img_c = img[a, b, :, :]
            
            # img_c = im_b.data.detach().numpy()
            img_c = img_c.detach().cpu().data.numpy()
            # img_c = img_c.data.cpu().detach().numpy()
            dst = cv2.cornerHarris(img_c, 3, 3, 0.04)
            dst = cv2.dilate(dst, None)
            img_c[dst > 0.7 * dst.max()] = [1]
            img_c[dst < 0.7 * dst.max()] = [0]
            img_c = torch.from_numpy(img_c)
            img_c = torch.unsqueeze(img_c, dim=0)
            att_1 = torch.cat((att_1, img_c), 0)
        att_2 = torch.unsqueeze(att_1, dim=0)
        att_0 = torch.cat((att_0, att_2), 0)

    return att_0

# 更改harries角点检测
def shi_tomasi(img):
    img = img.float()
    att_0 = torch.tensor([])
    for a in range(img.shape[0]):
        att_1 = torch.tensor([])
        for b in range(img.shape[1]):
            img_c = img[a, b, :, :]

            # 将灰度图像转换为PyTorch张量
            tensor = torch.unsqueeze(img_c, dim=0).unsqueeze(0)

            # 将张量移动到GPU上
            if torch.cuda.is_available():
                tensor = tensor.cuda()

            # 设置Shi-Tomasi角点检测参数
            max_corners = 100
            quality_level = 0.3
            min_distance = 7
            block_size = 7

            # 使用Shi-Tomasi角点检测算法检测角点
            corners = cv2.goodFeaturesToTrack(img_c.detach().cpu().numpy(), max_corners, quality_level, min_distance, blockSize=block_size)

            # 将角点标记在图像上
            for corner in corners:
                x, y = corner.ravel()
                img_c[int(np.round(y)), int(np.round(x))] = 1

            # 将处理后的图像转换为PyTorch张量
            tensor = torch.unsqueeze(img_c, dim=0).unsqueeze(0)

            # 将张量移回CPU上
            if torch.cuda.is_available():
                tensor = tensor.cpu()

            att_1 = torch.cat((att_1, tensor), 0)
        att_2 = torch.unsqueeze(att_1, dim=0)
        att_0 = torch.cat((att_0, att_2), 0)

    return att_0

# class COA(nn.Module):                                                                                                   #知识蒸馏+超分
#     def __init__(self, channel):
#         super(COA, self).__init__()
#         self.conv = nn.Sequential(
#                 nn.Conv2d(channel, channel, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         y = self.conv(x)
#         att2 = harris(x)
#         att_map = att2.to('cuda:0')
#         out = y + x
#         return out


# class COA(nn.Module):                                                                                                   #知识蒸馏+超分
#     def __init__(self, channel):
#         super(COA, self).__init__()
#         self.conv = nn.Sequential(
#                 nn.Conv2d(channel, channel, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         y = self.conv(x)
#         att2 = harris(x)
#         # att2 = shi_tomasi(x)
#         att_map = att2.to('cuda:2')
#         # out = y + x
#         x = x.to('cuda:2')
#         y = y.to('cuda:2')
#         out =  att_map*y  + x
#         out = out.to('cuda:1')
#         return out


class COA(nn.Module):                                                                                                   #知识蒸馏+超分
    def __init__(self, channel):
        super(COA, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y = self.conv(x)
        # att2 = harris(x)
        # att_map = att2.to('cuda:0')
        out = y + x
        return out



class SpatialAttention(nn.Module):
    def __init__(self, feature_size, head_num):
        super(SpatialAttention, self).__init__()

        self.feature_size = feature_size
        self.head_num = head_num

        self.conv = nn.ModuleList(
            [nn.Conv2d(feature_size // head_num, 1, 1, padding=0) for i in range(head_num)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_list = []
        att_list = []
        for i, x_ in enumerate(torch.chunk(x, self.head_num, 1)):
            att = self.conv[i](x_)
            att = self.sigmoid(att)
            att_list.append(att)
            x_list.append(x_ * att)

        att = torch.cat(att_list, 1)
        x = torch.cat(x_list, 1)

        return x,att

class DetailFeatureExtraction(nn.Module):
    def __init__(self, n_feat):
        super(DetailFeatureExtraction, self).__init__()
        self. num_ftrs = 64
        self. head_num = 8

        self.coa = COA(n_feat)
        self.spa = SpatialAttention(self.num_ftrs, self.head_num)                                                       #xilidu

        self.convp1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp2 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.conv1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(n_feat, n_feat, 1))

        self.conv4 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, stride=1, dilation=1, groups=1)
        self.conv_transpose = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        # channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 64, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feat1 = self.convp1(x)
        coa_feat = self.coa(feat1)
        coa_conv = self.convp2(coa_feat)
        coa_output = self.conv1(coa_conv)
#------------------------------------------------
        x3_max = self.maxpool(F.leaky_relu(self.conv4(x), negative_slope=0.2))

        x3_avg = self.avgpool(F.leaky_relu(self.conv4(x), negative_slope=0.2))
        x3 = torch.add(x3_max, x3_avg)
        # print(x3.shape)

        # channel attention
        x_gap = self.gap(x3)
        x_gap = x_gap.view(x_gap.size(0), -1)
        x_gap = self.relu(self.fc1(x_gap))
        x_gap = self.fc2(x_gap)
        x_weights = self.sigmoid(x_gap)
        x_weights = x_weights.view(x3.size(0), x3.size(1), 1, 1)
        # print(x_weights.shape)
        x3 = x3 * x_weights.expand_as(x3)
        x3 = self.conv_transpose(x3)
        #
        # print(coa_output.shape)
        # print(x3.shape)
        # spa_output, att = self.spa(x)
        #spa_output=torch.tensor(spa_output1)
        out = coa_output+x3
        #out = self.conv1(out)
        return out
# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 heads=[8, 8, 8],
                 ):

        super(Restormer_Encoder, self).__init__()

        self.num_ftrs = 64 * 1 * 1
        self.patch_embed = OverlapPatchEmbed(inp_channels, 3)
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction(n_feat = 64)
        self.ResNest = ResNest2(BottleNeck,[2, 2])
        
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU6(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU6(inplace=True), )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU6(inplace=True), )
        #self.spa = SpatialAttention(512, 8)
    def forward(self, inp_img):
        #print(inp_img.shape)
        #print('------------------')
        inp_enc_level1 = self.patch_embed(inp_img)
        # print(inp_img.shape)
        # out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.ResNest(inp_enc_level1)
        #out_enc_level1, att = self.spa(out_enc_level1)
        out_enc_level1 = self.conv1(out_enc_level1)
        out_enc_level1 = self.conv2(out_enc_level1)
        out_enc_level1 = self.conv3(out_enc_level1)
        #(out_enc_level1.shape)
        base_feature = self.baseFeature(out_enc_level1) # NAT
        detail_feature = self.detailFeature(out_enc_level1) # ASA
        #detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 #num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 #ffn_expansion_factor=2,
                 bias=False,
                 #LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(128, int(dim), kernel_size=1, bias=bias)
        #self.encoder_level2 = nn.Sequential(
        #   *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.ResNest = ResNest(BottleNeck, [2, 2], inchannel=32)
        self.output = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
        self.spa = SpatialAttention(128, 8)
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0, att = self.spa(out_enc_level0) # 元学习
        #print(out_enc_level0.shape)
        #out_enc_level0=base_feature+detail_feature
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level0 = out_enc_level0# .to('cuda:3')
        out_enc_level1 = self.ResNest(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)

        return self.sigmoid(out_enc_level1),att


# if __name__ == '__main__':
#
#     modelE = Restormer_Encoder()
#     base_feature, detail_feature, out_enc_level1 = modelE()
#     c = base_feature.shape[1] + detail_feature.shape[1]
#     modelD = Restormer_Decoder(c, 512, 32).cuda()



# if __name__ == '__main__':
#     import numpy as np
#     x = torch.DoubleTensor(np.random.rand(4, 1, 128, 128)).float()
#     modelE = Restormer_Encoder(1)

#     base_feature, detail_feature, out_enc_level1 = modelE(x)

#     c = base_feature.shape[1] + detail_feature.shape[1]
#     modelD = Restormer_Decoder(128, 512, 32)
#     out1, att= modelD(base_feature, detail_feature)

#     # device = torch.device('cuda:0')
#     # modelE = Restormer_Encoder(1).to(device)
#     # modelD = Restormer_Decoder(128, 512, 32).to(device)
#     print (out1.shape)


# import numpy as np
#
# if __name__ == '__main__':
#
#     y = torch.DoubleTensor(np.random.rand(2, 64, 12, 16)).float()
#     a = CSA(n_feat = 64)
#     x1,x2 = a(y)

