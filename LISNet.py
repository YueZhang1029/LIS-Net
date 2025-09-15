# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch.nn as nn
import torch.nn.functional as F
from networks.networks_other import init_weights
from networks.utils import UnetUp3, UnetUp3_CT,UnetUp3_CT2, UnetUp3_CT_AMM, UnetConv3
from networks.CMUNext_utils import CMUNext_block, conv_block, MSAG, ConvMixerBlock, MSAGCENTER9
from torchinfo import summary
from networks.DWT_IDWT import *
from networks.DWT_downsample import Downsample_L, Downsample_H, Upsample, Waveattention_block

class LISNet(nn.Module):

    def __init__(self, n_classes=4, is_deconv=True, in_channels=3, is_batchnorm=True, filters=[16, 32, 64, 128, 256], depth=7):
        super(LISNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        #
        # filters = [32, 64, 128, 256, 512]
        # kernels = [3, 3, 7, 7, 7]
        # depths = [1, 1, 1, 6, 3]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))#nn.Sequential 定义的只接受单输入

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center1 = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3,

            3), padding_size=(1, 1, 1))
        self.center2 = MSAGCENTER9(filters[4])
        # wavepool 基于小波分解的池化 H
        self.wavepool_H1 = nn.Sequential(*[Downsample_H(wavename='haar')])
        self.wavepool_H2 = nn.Sequential(*[Downsample_H(wavename='haar'), Downsample_H(wavename='haar')])
        self.wavepool_H3 = nn.Sequential(*[Downsample_H(wavename='haar'), Downsample_H(wavename='haar'), Downsample_H(wavename='haar')])
        self.wavepool_H4 = nn.Sequential(*[Downsample_H(wavename='haar'), Downsample_H(wavename='haar'), Downsample_H(wavename='haar'), Downsample_H(wavename='haar')])
        #LLL
        self.wavepool_L1 = nn.Sequential(*[Downsample_L(wavename='haar')])
        self.wavepool_L2 = nn.Sequential(*[Downsample_L(wavename='haar'), Downsample_L(wavename='haar')])
        self.wavepool_L3 = nn.Sequential(*[Downsample_L(wavename='haar'), Downsample_L(wavename='haar'), Downsample_L(wavename='haar')])
        self.wavepool_L4 = nn.Sequential(*[Downsample_L(wavename='haar'), Downsample_L(wavename='haar'), Downsample_L(wavename='haar'), Downsample_L(wavename='haar')])

        #FC 变换通道数
        self.fc1 = nn.Conv3d(self.in_channels, filters[1], kernel_size=1)
        self.fc2 = nn.Conv3d(self.in_channels, filters[2], kernel_size=1)
        self.fc3 = nn.Conv3d(self.in_channels, filters[3], kernel_size=1)
        self.fc4 = nn.Conv3d(self.in_channels, filters[4], kernel_size=1)

        # wavelet gate
        self.wave4 = Waveattention_block(filters[3])
        self.wave3 = Waveattention_block(filters[2])
        self.wave2 = Waveattention_block(filters[1])
        self.wave1 = Waveattention_block(filters[0])

        # upsampling1
        self.up_concat4_1 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3_1 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2_1 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1_1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # upsampling2
        self.up_concat4_2 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3_2 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2_2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1_2 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final2 = nn.Conv3d(filters[0], n_classes, 1)
        # dropout(1 center share 2 branches )
        self.dropout = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[16, 112, 112, 80]
        maxpool1 = self.maxpool1(conv1)#[16, 56, 56, 40]


        conv2 = self.conv2(maxpool1)#[32, 56, 56, 40]
        maxpool2 = self.maxpool2(conv2)#[32, 28, 28, 20]

        conv3 = self.conv3(maxpool2)#[64, 28, 28, 20]
        maxpool3 = self.maxpool3(conv3)#[64, 14, 14, 10]

        conv4 = self.conv4(maxpool3) # [128, 14, 14, 10]
        maxpool4 = self.maxpool4(conv4)#[128, 7, 7, 5]

        center1 = self.center1(maxpool4)#[256, 7, 7, 5]
        center = self.center2(center1)
        center = self.dropout(center)

        #wavepool L
        L1 = self.wavepool_L1(inputs)
        L1 = self.fc1(L1)
        L2 = self.wavepool_L2(inputs)
        L2 = self.fc2(L2)
        L3 = self.wavepool_L3(inputs)
        L3 = self.fc3(L3)
        # L4 = self.wavepool_L4(inputs)
        # L4 = self.fc4(L4)


        # H
        H1 = self.wavepool_H1(inputs)
        H1 = self.fc1(H1)
        H2 = self.wavepool_H2(inputs)
        H2 = self.fc2(H2)
        H3 = self.wavepool_H3(inputs)
        H3 = self.fc3(H3)
        # H4 = self.wavepool_H4(inputs)
        # H4 = self.fc4(H4)

        #concat
        # L4 = self.wave4(L4, conv4)
        L3 = self.wave4(L3, conv4)
        L2 = self.wave3(L2, conv3)
        L1 = self.wave2(L1, conv2)

        # H4 = self.wave4(H4, conv4)
        H3 = self.wave4(H3, conv4)
        H2 = self.wave3(H2, conv3)
        H1 = self.wave2(H1, conv2)

        up4_1 = self.up_concat4_1(H3, center)
        up3_1 = self.up_concat3_1(H2, up4_1)
        up2_1 = self.up_concat2_1(H1, up3_1)
        up1_1 = self.up_concat1_1(conv1, up2_1)
        up1 = self.dropout1(up1_1)
        out_1 = self.final1(up1)

        # decoder2
        up4_2 = self.up_concat4_2(L3, center)#[128, 14, 14, 10] [256, 7, 7, 5]-->center up:256,14,14,10 concat:384,14,14,10 -->conv:128,14,14,10
        up3_2 = self.up_concat3_2(L2, up4_2)#[64, 28, 28, 20] [128, 14, 14, 10]
        up2_2 = self.up_concat2_2(L1, up3_2)
        up1_2 = self.up_concat1_2(conv1, up2_2)
        up2 = self.dropout2(up1_2)
        out_2 = self.final2(up2)

        return out_1, out_2

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = LISNet(n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True, filters=[16, 32, 64, 128, 256])
    summary(model, (4, 1, 112, 112, 80), device='cuda')
    print(model)