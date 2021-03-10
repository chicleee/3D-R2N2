# -*- coding: utf-8 -*-

import paddle
import paddle
import paddle.nn.functional as F
import numpy as np

class Residual_Block(paddle.nn.Layer):
    def __init__(self, in_c, out_c, ks = 3, stride_ = 1, pad = 1, neg_slope = 0.1, shortcut = None):
        super(Residual_Block, self).__init__()
        self.path = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels = in_c, out_channels = out_c, kernel_size = ks, stride = stride_, padding = pad, bias = False),
            paddle.nn.BatchNorm2D(num_features = out_c),
            paddle.nn.LeakyReLU(negative_slope = neg_slope, inplace = True),
            paddle.nn.Conv2D(in_channels = out_c, out_channels = out_c, kernel_size = 3, stride = 1, padding = 1, bias = False),
            paddle.nn.BatchNorm2D(num_features = out_c)
        )
        self.shortcut = shortcut
        self.out_relu = paddle.nn.LeakyReLU(negative_slope = neg_slope, inplace = True)

    def forward(self, x):
        out = self.path(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return self.out_relu(out)

# class FCConv3D(paddle.nn.Layer):
#     def __init__(self, in_f, filter_shape, out_shape, seq_len):
#         # filter_shape = (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
#         self.fc = nn.Linear(in_f, int(np.prod(out_shape[1:])), bias = False)
#         self.Conv3D = nn.Conv3D(filter_shape[0], filter_shape[1], 
#             kernel_size = filter_shape[2], padding = int((filter_shape[2] - 1) / 2), bias = False)
#         self.bn1 = Recurrent_BatchNorm3d(filter_shape[0], seq_len)
#         self.bn2 = Recurrent_BatchNorm3d(filter_shape[0], seq_len)
#         self.bias = nn.Parameter(torch.FloatTensor(1, out_shape[1], 1, 1, 1).fill_(0.1))

#     def forward(self, feature_x, h_t, idx):
#         feature_x = self.fc(feature_x).view(*self.out_shape)
#         feature_x = self.bn1(feature_x, idx)
#         h_t = self.Conv3D(h_t)
#         h_t = self.bn2(h_t, idx)
#         out = feature_x + h_t + self.bias
#         return out

class Recurrence_FCConv3D(paddle.nn.Layer):
    def __init__(self, in_features, filters, n_gru_vox):
        super(Recurrence_FCConv3D, self).__init__()
        self.out_shape = [-1, filters, n_gru_vox, n_gru_vox, n_gru_vox]
        self.fc = nn.Linear(in_features, int(np.prod(self.out_shape[1:])))
        self.Conv3D = nn.Conv3D(filters, filters, [3, 3, 3], padding = 1)
        self.Conv3D_rh = nn.Sequential(
            nn.Conv3D(filters, filters, [3, 3, 3], padding = 1),
            nn.ReLU()
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, feature_x, h_t):
        feature_x = self.fc(feature_x).view(*self.out_shape)
        out = self.Conv3D(h_t)
        u_t = self.sigmoid1(out)
        r_t = self.sigmoid2(out)
        h_next = (1.0 - u_t) * h_t + u_t * self.tanh(self.Conv3D_rh(r_t * h_t))
        return h_next

class Recurrent_BatchNorm3d(paddle.nn.Layer):
    def __init__(self, num_features, T_max, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True):
        super(Recurrent_BatchNorm3d, self).__init__()
        self.num_features = num_features
        self.T_max = T_max
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # Has learnable affine parameters
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        for i in range(self.T_max):
            self.register_buffer('running_mean_{}'.format(i), torch.zeros(self.num_features) if self.track_running_stats else None)
            self.register_buffer('running_var_{}'.format(i), torch.zeros(self.num_features) if self.track_running_stats else None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            for i in range(self.T_max):
                running_mean = getattr(self, 'running_mean_{}'.format(i))
                running_var = getattr(self, 'running_var_{}'.format(i))

                running_mean.zero_()
                running_var.fill_(1)

        if self.affine:
            #according to the paper, 0.1 is a good initialization for gamma
            self.weight.data.fill_(0.1)
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input_.dim()))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.T_max:
            time = self.T_max - 1

        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))

        return nn.functional.batch_norm(input_, \
            running_mean = running_mean, \
            running_var = running_var, \
            weight = self.weight, \
            bias = self.bias, \
            training = self.training, \
            momentum = self.momentum, \
            eps = self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
            ' T_max={T_max}, affine={affine})'
            .format(name=self.__class__.__name__, **self.__dict__))

class BN_FCConv3D(paddle.nn.Layer):
    def __init__(self, fc_w_fan_in, filter_shape, deconv_filter, n_gru_vox, seq_len):
        super(BN_FCConv3D, self).__init__()
        self.deconv_filter = deconv_filter
        self.n_gru_vox = n_gru_vox
        self.fc = nn.Linear(fc_w_fan_in, deconv_filter * n_gru_vox * n_gru_vox * n_gru_vox, bias = False)
        self.Conv3D = nn.Conv3D(filter_shape[0], filter_shape[1], \
                                kernel_size = filter_shape[2], \
                                padding = int((filter_shape[2] - 1) / 2), bias = False)
        self.bn1 = Recurrent_BatchNorm3d(filter_shape[0], seq_len)
        self.bn2 = Recurrent_BatchNorm3d(filter_shape[0], seq_len)
        self.bias = nn.Parameter(torch.FloatTensor(1, deconv_filter, 1, 1, 1).fill_(0.1))

    def forward(self, x, h, idx):
        x = self.fc(x).view(-1, self.deconv_filter, self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        bn_x = self.bn1(x, idx)
        Conv3D = self.Conv3D(h)
        bn_Conv3D = self.bn2(Conv3D, idx)
        return bn_x + bn_Conv3D + self.bias

class Unpool3D(paddle.nn.Layer):
    def __init__(self, unpool_size = 2, padding = 0):
        super(Unpool3D, self).__init__()
        self.unpool_size = unpool_size
        self.padding = padding

    def forward(self, x):
        out_shape = (x.size(0), x.size(1), 
            self.unpool_size * x.size(2), self.unpool_size * x.size(3), self.unpool_size * x.size(4))
        out = torch.FloatTensor(*out_shape).zero_()

        if torch.cuda.is_available():
            out = out.type(torch.cuda.FloatTensor)

        out = torch.autograd.Variable(out)
        out[:, \
            :, \
            self.padding : self.padding + out_shape[2] + 1 : self.unpool_size, \
            self.padding : self.padding + out_shape[3] + 1 : self.unpool_size, \
            self.padding : self.padding + out_shape[4] + 1 : self.unpool_size] = x
        return out

class SoftmaxLoss3D(paddle.nn.Layer):
    def __init__(self):
        super(SoftmaxLoss3D, self).__init__()

    def forward(self, inputs, gt):
        max_channel = paddle.max(inputs, dim = 1, keepdim = True)[0]
        adj_inputs = inputs - max_channel
        exp_x = paddle.exp(adj_inputs)
        sum_exp_x = paddle.sum(exp_x, dim = 1, keepdim = True)

        loss = paddle.mean(
            paddle.sum(-gt * adj_inputs, dim = 1, keepdim = True) + \
            paddle.log(sum_exp_x)
        )

        return loss
