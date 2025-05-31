""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""
# -*-coding:utf-8 -*-
import torch.nn.functional as F
from .unet_parts import *
import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.double_conv(x)
        out += identity
        out = self.relu(out)
        return out

class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(ResUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Further reduce the number of channels
        self.inc = ResBlock(n_channels, 16)
        self.down1 = ResDown(16, 32)
        self.down2 = ResDown(32, 64)
        self.down3 = ResDown(64, 128)
        self.down4 = ResDown(128, 128)
        self.up1 = ResUp(256, 64, bilinear)
        self.up2 = ResUp(128, 32, bilinear)
        self.up3 = ResUp(64, 16, bilinear)
        self.up4 = ResUp(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits