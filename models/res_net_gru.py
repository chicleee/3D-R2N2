# -*- coding: utf-8 -*-

import paddle
import paddle.nn.functional as F
import math
from models.layer import Residual_Block, BN_FCConv3D, Unpool3D
# from torch.utils.checkpoint import checkpoint

class Encoder(paddle.nn.Layer):
    def __init__(self, seq_len, n_gru_vox, deconv_filters):
        super(Encoder, self).__init__()

        # Parameters
        self.conv_filters = [3, 96, 128, 256, 256, 256, 256]
        self.fc_layers_size = [1024]
        self.ceil_mode = True
        self.img_h, self.img_w = 127, 127
        self.n_gru_vox = n_gru_vox
        self.deconv_filters = deconv_filters
        self.seq_len = seq_len

        # Build the network
        self.build()

    def build(self):
        '''
            Encoder
        '''
        # layer 1
        self.layer1 = Residual_Block(self.conv_filters[0], self.conv_filters[1], ks = 7, pad = 3)
        self.layer1_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 2
        self.layer2_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[1], self.conv_filters[2], kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[2])
        )
        self.layer2 = Residual_Block(self.conv_filters[1], self.conv_filters[2], shortcut = self.layer2_sc)
        self.layer2_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 3
        self.layer3_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[2], self.conv_filters[3], kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[3])
        )
        self.layer3 = Residual_Block(self.conv_filters[2], self.conv_filters[3], shortcut = self.layer3_sc)
        self.layer3_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 4
        self.layer4 = Residual_Block(self.conv_filters[3], self.conv_filters[4])
        self.layer4_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 5
        self.layer5_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[4], self.conv_filters[5], kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[5])
        )
        self.layer5 = Residual_Block(self.conv_filters[4], self.conv_filters[5], shortcut = self.layer5_sc)
        self.layer5_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 6
        self.layer6_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[5], self.conv_filters[6], kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[6])
        )
        self.layer6 = Residual_Block(self.conv_filters[5], self.conv_filters[6], shortcut = self.layer6_sc)
        self.layer6_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # final layer size
        fh, fw = self.fm_size()
        fcs_size = fh * fw * self.conv_filters[-1]

        # fc layers
        self.fcs = paddle.nn.Linear(fcs_size, self.fc_layers_size[0])


        '''
            GRU3d
        '''
        Conv3D_filter_shape = (self.deconv_filters[0], self.deconv_filters[0], 3, 3, 3)
        self.gru3d_u = BN_FCConv3D(self.fc_layers_size[-1], \
            Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
        self.gru3d_r = BN_FCConv3D(self.fc_layers_size[-1], \
            Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
        self.gru3d_rs = BN_FCConv3D(self.fc_layers_size[-1], \
            Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
        self.gru3d_sigmoid = paddle.nn.Sigmoid()
        self.gru3d_tanh = paddle.nn.Tanh()

    def fm_size(self):
        h = self.img_h
        w = self.img_w
        for i in range(len(self.conv_filters) - 1):
            if self.ceil_mode is True:
                h = math.ceil(h / 2)
                w = math.ceil(w / 2)
            else:
                h = math.floor(h / 2)
                w = math.floor(w / 2)
        return int(h), int(w)

    def forward(self, x, h, idx):
        bs = x.size()[0]
        # encoder
        x = self.layer1(x)
        x = self.layer1_pool(x)
        x = self.layer2(x)
        x = self.layer2_pool(x)
        x = self.layer3(x)
        x = self.layer3_pool(x)
        x = self.layer4(x)
        x = self.layer4_pool(x)
        x = self.layer5(x)
        x = self.layer5_pool(x)
        x = self.layer6(x)
        x = self.layer6_pool(x)
        x = x.view(bs, -1)
        x = self.fcs(x)

        # gru
        update = self.gru3d_sigmoid(self.gru3d_u(x, h, idx))
        reset = self.gru3d_sigmoid(self.gru3d_r(x, h, idx))
        rs = self.gru3d_tanh(self.gru3d_rs(x, reset * h, idx))
        x = update * h + (1.0 - update) * rs

        return x, update


class Decoder(paddle.nn.Layer):
    def __init__(self, deconv_filters):
        super(Decoder, self).__init__()

        # Parameter
        self.deconv_filters = deconv_filters

        self.build()

    def build(self):
        self.decoder_unpool0 = paddle.nn.ConvTranspose3d(self.deconv_filters[0], self.deconv_filters[0], kernel_size = 2, stride = 2)
        self.decoder_block0 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[0], self.deconv_filters[1], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[1]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[1], self.deconv_filters[1], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[1])
        )

        self.decoder_unpool1 = paddle.nn.ConvTranspose3d(self.deconv_filters[1], self.deconv_filters[1], kernel_size = 2, stride = 2)
        self.decoder_block1 = paddle.nn.Sequential(
            paddle.nn.Conv3D(self.deconv_filters[1], self.deconv_filters[2], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[2]),
            paddle.nn.LeakyReLU(negative_slope = 0.1),
            paddle.nn.Conv3D(self.deconv_filters[2], self.deconv_filters[2], 3, padding = 1),
            paddle.nn.BatchNorm3D(self.deconv_filters[2])
        )

        self.decoder_unpool2 = paddle.nn.ConvTranspose3d(self.deconv_filters[2], self.deconv_filters[2], kernel_size = 2, stride = 2)
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
            paddle.nn.Conv3D(self.deconv_filters[4], self.deconv_filters[5], 3, padding = 1),
            paddle.nn.LeakyReLU(negative_slope = 0.1)
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

        return x


class Res_Gru_Net(paddle.nn.Layer):
    def __init__(self, seq_len):
        super(Res_Gru_Net, self).__init__()

        self.deconv_filters = [128, 128, 128, 64, 32, 2]
        self.n_gru_vox = 4
        self.seq_len = seq_len
        self.encoder = Encoder(seq_len, self.n_gru_vox, self.deconv_filters)
        self.decoder = Decoder(self.deconv_filters)

    def hidden_init(self, shape):
        h = paddle.zeros(shape, dtype='float32')
        return h

    def forward(self, x):
        '''
            x: [bs, seq_len, c, h, w]
        '''
        # encoder
        bs, seq_len = x.size()[:2]
        h_shape = [bs, self.deconv_filters[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox]
        h = self.hidden_init(h_shape)

        for idx in range(self.seq_len):
            h, u = self.encoder(x[:, idx, ...], h, idx)

        # decoder
        h = self.decoder(h)
        return h

def weight_init(network):
    for each_module in network.modules():
        if isinstance(each_module, (paddle.nn.Conv2D, paddle.nn.Conv3D)):
            paddle.nn.init.xavier_uniform_(each_module.weight)
            if each_module.bias is not None:
                each_module.bias.data.zero_()
        elif isinstance(each_module, (paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
            each_module.weight.data.fill_(1.)
            if each_module.bias is not None:
                each_module.bias.data.zero_()
        elif isinstance(each_module, paddle.nn.Linear):
            each_module.weight.data.normal_(0, 0.01)
            if each_module.bias is not None:
                each_module.bias.data.zero_()

if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    model = paddle.Model(Res_Gru_Net(cfg))
    model.summary((16, 5, 3, 127, 127))