# -*- coding: utf-8 -*-

import paddle
import paddle.nn.functional as F
import numpy as np
import math
import warnings

class Residual_Block(paddle.nn.Layer):
    def __init__(self, in_c, out_c, ks = 3, stride_ = 1, pad = 1, neg_slope = 0.1, shortcut = None):
        super(Residual_Block, self).__init__()
        self.path = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels = in_c, out_channels = out_c, kernel_size = ks, stride = stride_, padding = pad, weight_attr=paddle.nn.initializer.XavierUniform(), bias_attr = False),
            paddle.nn.BatchNorm2D(num_features = out_c),
            paddle.nn.LeakyReLU(negative_slope = neg_slope),
            paddle.nn.Conv2D(in_channels = out_c, out_channels = out_c, kernel_size = 3, stride = 1, padding = 1, weight_attr=paddle.nn.initializer.XavierUniform(), bias_attr = False),
            paddle.nn.BatchNorm2D(num_features = out_c)
        )
        self.shortcut = shortcut
        self.out_relu = paddle.nn.LeakyReLU(negative_slope = neg_slope)

    def forward(self, x):
        out = self.path(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return self.out_relu(out)

# reference--#
# https://github.com/PaddlePaddle/Paddle/blob/6a19e41f1faa480f2a5000e5b5a4e199bed4fc1a/python/paddle/nn/layer/norm.py#L63
# https://github.com/Amaranth819/3D-R2N2-Pytorch/blob/master/lib/layer.py
class Recurrent_BatchNorm3D(paddle.nn.Layer):
    def __init__(self,
                 num_features,
                 T_max,
                 momentum=0.1,
                 epsilon=1e-5,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 name=None):
        super(Recurrent_BatchNorm3D, self).__init__()
        self._num_features = num_features
        self.T_max = T_max
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        if paddle.get_default_dtype() == 'float16':
            paddle.set_default_dtype('float32')

        param_shape = [num_features]

        # create parameter
        if weight_attr == False:
            self.weight = self.create_parameter(
                attr=None, shape=param_shape, default_initializer=paddle.nn.initializer.Constant(0.1))
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=paddle.nn.initializer.Constant(0.1))
            self.weight.stop_gradient = self._weight_attr != None and self._weight_attr.learning_rate == 0.

        if bias_attr == False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                default_initializer=paddle.nn.initializer.Constant(0.0),
                is_bias=True)
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True)
            self.bias.stop_gradient = self._bias_attr != None and self._bias_attr.learning_rate == 0.

        # moving_mean_name = None
        # moving_variance_name = None

        for i in range(self.T_max):
            # if name is not None:
            # moving_mean_name = 'running_mean_{}'.format(i)
            # moving_variance_name = 'running_var_{}'.format(i)
            # self._mean = self.create_parameter(
            #     attr=paddle.ParamAttr(
            #         name=moving_mean_name,
            #         initializer=paddle.nn.initializer.Constant(0.0),
            #         trainable=False,
            #         do_model_average=True),
            #     shape=param_shape)
            # self._mean.stop_gradient = True

            # self._variance = self.create_parameter(
            #     attr=paddle.ParamAttr(
            #         name=moving_variance_name,
            #         initializer=paddle.nn.initializer.Constant(1.0),
            #         trainable=False,
            #         do_model_average=True),
            #     shape=param_shape)
            # self._variance.stop_gradient = True

            self.register_buffer('running_mean_{}'.format(i), paddle.zeros(param_shape))
            self.register_buffer('running_var_{}'.format(i), paddle.zeros(param_shape))
        
        self.reset_parameters()


        self._data_format = data_format
        self._in_place = False
        self._momentum = momentum
        self._epsilon = epsilon
        self._fuse_with_relu = False
        self._name = name

    def reset_parameters(self):
        for i in range(self.T_max):
            running_mean = getattr(self, 'running_mean_{}'.format(i))
            running_var = getattr(self, 'running_var_{}'.format(i))
            # print(running_mean)
            # running_mean.zero_()
            # running_var.fill_(1)
            running_mean = paddle.zeros_like(running_mean, dtype='float32')
            running_var = paddle.full_like(running_var, fill_value=1., dtype='float32')

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NCDHW':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == "NDHWC":
            self._data_format = 'NHWC'
        else:
            raise ValueError(
                'expected NCDHW, NDHWC or None for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                len(input.shape)))

    def forward(self, input, time):

        self._check_data_format(self._data_format)

        self._check_input_dim(input)

        if time >= self.T_max:
            time = self.T_max - 1

        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))

        if self.training:
            warnings.warn(
                "When training, we now always track global mean and variance.")

        return F.batch_norm(
            input,
            # self._mean,
            # self._variance,
            running_mean,
            running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format,)

    def extra_repr(self):
        main_str = 'num_features={}, momentum={}, epsilon={}'.format(
            self._num_features, self._momentum, self._epsilon)
        if self._data_format is not 'NCHW':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str


class BN_FCConv3D(paddle.nn.Layer):
    def __init__(self, fc_w_fan_in, filter_shape, deconv_filter, n_gru_vox, seq_len):
        super(BN_FCConv3D, self).__init__()
        self.deconv_filter = deconv_filter
        self.n_gru_vox = n_gru_vox
        self.fc = paddle.nn.Linear(fc_w_fan_in, deconv_filter * n_gru_vox * n_gru_vox * n_gru_vox, bias_attr = False)
        self.conv3d = paddle.nn.Conv3D(filter_shape[0], filter_shape[1], \
                                kernel_size = filter_shape[2], \
                                 weight_attr=paddle.nn.initializer.XavierUniform(), \
                                padding = int((filter_shape[2] - 1) / 2), bias_attr = False)
        self.bn1 = Recurrent_BatchNorm3D(filter_shape[0], seq_len)
        self.bn2 = Recurrent_BatchNorm3D(filter_shape[0], seq_len)
        self.bias = paddle.full(shape=[1, deconv_filter, 1, 1, 1], fill_value=0.1, dtype='float32')

    def forward(self, x, h, idx):
        x = paddle.reshape(self.fc(x), [-1, self.deconv_filter, self.n_gru_vox, self.n_gru_vox, self.n_gru_vox])
        # print("idx", idx)
        # print("x.shape", x.shape)
        # print("h.shape", h.shape)

        bn_x = self.bn1(x, idx)
        conv3d = self.conv3d(h)
        bn_conv3d = self.bn2(conv3d, idx)

        return bn_x + bn_conv3d + self.bias

class Encoder(paddle.nn.Layer):
    def __init__(self, cfg, n_gru_vox, deconv_filters):
        super(Encoder, self).__init__()

        # Parameters
        self.conv_filters = [3, 96, 128, 256, 256, 256, 256]
        self.fc_layers_size = [1024]
        self.ceil_mode = True
        self.img_h, self.img_w = 127, 127
        self.n_gru_vox = n_gru_vox
        self.deconv_filters = deconv_filters
        # self.cfg = cfg
        self.seq_len = cfg.CONST.N_VIEWS_RENDERING

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
            paddle.nn.Conv2D(self.conv_filters[1], self.conv_filters[2], weight_attr=paddle.nn.initializer.XavierUniform(), kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[2])
        )
        self.layer2 = Residual_Block(self.conv_filters[1], self.conv_filters[2], shortcut = self.layer2_sc)
        self.layer2_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 3
        self.layer3_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[2], self.conv_filters[3], weight_attr=paddle.nn.initializer.XavierUniform(), kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[3])
        )
        self.layer3 = Residual_Block(self.conv_filters[2], self.conv_filters[3], shortcut = self.layer3_sc)
        self.layer3_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 4
        self.layer4 = Residual_Block(self.conv_filters[3], self.conv_filters[4])
        self.layer4_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 5
        self.layer5_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[4], self.conv_filters[5], weight_attr=paddle.nn.initializer.XavierUniform(), kernel_size = 1),
            paddle.nn.BatchNorm2D(self.conv_filters[5])
        )
        self.layer5 = Residual_Block(self.conv_filters[4], self.conv_filters[5], shortcut = self.layer5_sc)
        self.layer5_pool = paddle.nn.MaxPool2D(2, ceil_mode = self.ceil_mode)

        # layer 6
        self.layer6_sc = paddle.nn.Sequential(
            paddle.nn.Conv2D(self.conv_filters[5], self.conv_filters[6], weight_attr=paddle.nn.initializer.XavierUniform(), kernel_size = 1),
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
        self.gru3d_u = BN_FCConv3D(self.fc_layers_size[-1], Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
        self.gru3d_r = BN_FCConv3D(self.fc_layers_size[-1], Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
        self.gru3d_rs = BN_FCConv3D(self.fc_layers_size[-1], Conv3D_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
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

    def forward(self, x, h, u, idx):
        bs = x.shape[0]
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
        x = paddle.reshape(x, [bs, -1])
        x = self.fcs(x)

        # gru
        update = self.gru3d_sigmoid(self.gru3d_u(x, h, idx))
        reset = self.gru3d_sigmoid(self.gru3d_r(x, h, idx))
        rs = self.gru3d_tanh(self.gru3d_rs(x, reset * h, idx))
        x = update * h + (1.0 - update) * rs

        return x, update
