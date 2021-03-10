# -*- coding: utf-8 -*-

import paddle
import paddle.nn.functional as F
import numpy as np
import math
import warnings

class Decoder(paddle.nn.Layer):
    def __init__(self, deconv_filters):
        super(Decoder, self).__init__()

        # Parameter
        self.deconv_filters = deconv_filters

        self.build()

    def build(self):
        self.decoder_unpool0 = paddle.nn.Conv3DTranspose(self.deconv_filters[0], self.deconv_filters[0], kernel_size = 2, stride = 2)
        self.decoder_block0 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[0], self.deconv_filters[1], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[1]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[1], self.deconv_filters[1], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[1])
        )

        self.decoder_unpool1 = paddle.nn.Conv3DTranspose(self.deconv_filters[1], self.deconv_filters[1], kernel_size = 2, stride = 2)
        self.decoder_block1 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[1], self.deconv_filters[2], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[2]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[2], self.deconv_filters[2], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[2])
        )

        self.decoder_unpool2 = paddle.nn.Conv3DTranspose(self.deconv_filters[2], self.deconv_filters[2], kernel_size = 2, stride = 2)
        self.decoder_block2 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[2], self.deconv_filters[3], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[3]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[3], self.deconv_filters[3], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[3])
        )
        self.decoder_block2_shortcut = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[2], self.deconv_filters[3], 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[3])
        )

        self.decoder_block3 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[3], self.deconv_filters[4], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[4]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[4], self.deconv_filters[4], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[4]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[4], self.deconv_filters[4], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[4])
        )
        self.decoder_block3_shortcut = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[3], self.deconv_filters[4], 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[4])
        )
        
        self.decoder_block4 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[4], 1, 3, padding = 1),
            # paddle.nn.LeakyReLU(negative_slope = 0.1)
            paddle.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_unpool0(x)
        p = self.decoder_block0(x)
        x = F.leaky_relu(x + p)

        x = self.decoder_unpool1(x)
        p = self.decoder_block1(x)
        x = F.leaky_relu(x + p)

        x = self.decoder_unpool2(x)
        p1 = self.decoder_block2(x)
        p2 = self.decoder_block2_shortcut(x)
        x = F.leaky_relu(p1 + p2)

        p1 = self.decoder_block3(x)
        p2 = self.decoder_block3_shortcut(x)
        x = F.leaky_relu(p1 + p2)

        x = self.decoder_block4(x)

        # x = x[:,0,:,:,:]
        # x = F.softmax(x, axis=1)
        x = paddle.sum(x, axis=1)
        x = paddle.clip(x, min=0, max=1)

        return x 