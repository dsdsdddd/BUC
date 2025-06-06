import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Monodepth2Encoder(nn.Module):
    def __init__(self):
        super(Monodepth2Encoder, self).__init__()
        resnet = models.resnet18(weights=None)  # 可替换为预训练权重
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class Monodepth2Decoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(Monodepth2Decoder, self).__init__()
        self.upconv4 = ConvBlock(num_ch_enc[3], 256)
        self.upconv3 = ConvBlock(256 + num_ch_enc[2], 128)
        self.upconv2 = ConvBlock(128 + num_ch_enc[1], 64)
        self.upconv1 = ConvBlock(64 + num_ch_enc[0], 32)
        self.dispconv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, features):
        x4, x3, x2, x1 = features[::-1]  # reverse order
        up4 = F.interpolate(self.upconv4(x4), scale_factor=2)
        up3 = F.interpolate(self.upconv3(torch.cat([up4, x3], dim=1)), scale_factor=2)
        up2 = F.interpolate(self.upconv2(torch.cat([up3, x2], dim=1)), scale_factor=2)
        up1 = F.interpolate(self.upconv1(torch.cat([up2, x1], dim=1)), scale_factor=2)
        disp = torch.sigmoid(self.dispconv(up1)) * 10  # 深度范围可调
        return disp


class Monodepth2(nn.Module):
    def __init__(self):
        super(Monodepth2, self).__init__()
        self.encoder = Monodepth2Encoder()
        self.decoder = Monodepth2Decoder(num_ch_enc=[64, 64, 128, 256])

    def forward(self, x):
        features = self.encoder(x)
        disp = self.decoder(features)
        return disp
