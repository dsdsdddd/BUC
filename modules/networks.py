import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F

from modules.layers import BasicBlock
from utils.generic_utils import upsample
from modules.monodepth2 import Monodepth2 


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers

'''
class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BayesianConv2d, self).__init__()

        # 定义权重参数的先验分布
        self.weight_prior_mean = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        self.weight_prior_logvar = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)

        # 定义偏置项的先验分布
        self.bias_prior_mean = torch.zeros(out_channels)
        self.bias_prior_logvar = torch.zeros(out_channels)

        # 定义权重参数和偏置项的后验分布参数
        self.weight_post_mean = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_post_logvar = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_post_mean = nn.Parameter(torch.randn(out_channels))
        self.bias_post_logvar = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # 从权重参数的后验分布中采样权重
        weight = self.weight_post_mean + torch.exp(0.5 * self.weight_post_logvar) * torch.randn_like(self.weight_post_mean)
        # 从偏置项的后验分布中采样偏置
        bias = self.bias_post_mean + torch.exp(0.5 * self.bias_post_logvar) * torch.randn_like(self.bias_post_mean)
        # 进行卷积操作
        output = F.conv2d(x, weight, bias, padding=1)
        return output'''


class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BayesianConv2d, self).__init__()

        # 使用Bayesian层，例如BayesianLinear或BayesianConv2d
        self.conv_mean = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_logvar = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # 前向传播中同时返回均值和对数方差
        mean = self.conv_mean(x)
        logvar = self.conv_logvar(x)
        # 在通道维度上合并均值和对数方差
        combined_output = torch.cat([mean, logvar], dim=1)
        return combined_output

class UncertaintyEstimator(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels=1, use_refl=True):
        super().__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = BayesianConv2d(in_channels, out_channels, 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)

        return out

class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                scales=range(4), 
                num_output_channels=1,  
                use_skips=True
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                self.convs[f"output_{i}"] = nn.Sequential(
                BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                nn.Conv2d(num_ch_out, self.num_output_channels, 1),
                )

        
        for i in range(4):
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"uncert_conv_{i}"] = UncertaintyEstimator(num_ch_out,1)  #这个定义真的合理吗，输出是概率吗'''

        self.monodepth = Monodepth2()

    def gaussian_fusion(self, mv_depth, mono_depth, mask, sigma=0.5):
        weight_mv = torch.exp(- (mv_depth - mono_depth) ** 2 / (2 * sigma ** 2))
        weight_mono = 1 - weight_mv
        fused = weight_mv * mv_depth + weight_mono * mono_depth
        return fused * mask + mv_depth * (1 - mask)
    
    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]

                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]

                depth_outputs[f"uncert_s{i}_b1hw"] = self.convs[f"uncert_conv_{i}"](output)
                depth_outputs[f"log_depth_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)
                #print("depth_outputs[uncert_s{i}_b1hw]:",depth_outputs[f"uncert_s{i}_b1hw"].size(),depth_outputs[f"uncert_s{i}_b1hw"])
                #print(depth_outputs[f"log_depth_pred_s{i}_b1hw"])

            prev_outputs = outputs[::-1]

         if target_img is not None and mv_depth is not None and dynamic_mask is not None:
            mono_depth = self.monodepth(target_img)
            fused_depth = self.gaussian_fusion(mv_depth, mono_depth, dynamic_mask)
            depth_outputs['fused_depth'] = fused_depth

        return depth_outputs



class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1 if i == 0 else 2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)     
        return outputs

class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation = False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(
                self, 
                num_layers, 
                num_ch_out, 
                pretrained=True,
                antialiased=True,
            ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])

        model_source = antialiased_cnns if antialiased else models
        resnets = {18: model_source.resnet18,
                   34: model_source.resnet34,
                   50: model_source.resnet50,
                   101: model_source.resnet101,
                   152: model_source.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        encoder = resnets[num_layers](pretrained)

        resnet_backbone = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
        ]

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_out = num_ch_out

        self.net = nn.Sequential(
            *resnet_backbone,
            nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128, 
                    self.num_ch_out, 
                    (3, 3), 
                    padding=1, 
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )

    def forward(self, input_image):
        return self.net(input_image)

class UNetMatchingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
                                        "mnasnet_100", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        self.decoder = FeaturePyramidNetwork(
                                        self.encoder.feature_info.channels(), 
                                        out_channels=32,
                                    )
        self.outconv = nn.Sequential(
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(32, 16, 1),
                                    nn.InstanceNorm2d(16),
                                )

    def forward(self, x):
        encoder_feats = {f"feat_{i}": f for i, f in enumerate(self.encoder(x))}
        return self.outconv(self.decoder(encoder_feats)["feat_1"])
    

