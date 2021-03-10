# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/aistudio/work/3D-R2N2/models")
import paddle
import paddle.nn
import paddle.nn.functional as F
import numpy as np
import math
from encoder import Encoder
from decoder import Decoder

class Res_Gru_Net(paddle.nn.Layer):
    def __init__(self, cfg):
        super(Res_Gru_Net, self).__init__()

        self.deconv_filters = [128, 128, 128, 64, 32, 2]
        self.n_gru_vox = 4
        self.cfg = cfg
        self.encoder = Encoder(cfg, self.n_gru_vox, self.deconv_filters)
        self.decoder = Decoder(self.deconv_filters)

    def hidden_init(self, shape):
        h = paddle.zeros(shape, dtype='float32')
        return h

    def forward(self, rendering_images):
        '''
            x: [bs, seq_len, c, h, w]
        '''
        # encoder
        # [batch_size, n_views, img_c, img_h, img_w]
        bs, seq_len = rendering_images.shape[:2]
        # print(bs, seq_len) # 16 5
        h_shape = [bs, self.deconv_filters[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox] # [16, 128, 4, 4, 4]]
        h = self.hidden_init(h_shape)
        u = self.hidden_init(h_shape)

        rendering_images = paddle.transpose(rendering_images, perm=[1, 0, 2, 3, 4]) 
        # [n_views, batch_size, img_c, img_h, img_w]
        rendering_images = paddle.split(rendering_images, num_or_sections=rendering_images.shape[0], axis=0) 
        # image_features = []

        for idx, img in enumerate(rendering_images):
            features = paddle.squeeze(img, axis=0)
            # [batch_size, img_c, img_h, img_w]
            h, u = self.encoder(features, h, u, idx)

        # # decoder
        h = self.decoder(h)
        return h

# def weight_init(network):
#     for each_module in network.modules():
#         if isinstance(each_module, (paddle.nn.Conv2d, paddle.nn.Conv3D)):
#             paddle.nn.init.xavier_uniform_(each_module.weight)
#             if each_module.bias is not None:
#                 each_module.bias.data.zero_()
#         elif isinstance(each_module, (paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
#             each_module.weight.data.fill_(1.)
#             if each_module.bias is not None:
#                 each_module.bias.data.zero_()
#         elif isinstance(each_module, paddle.nn.Linear):
#             each_module.weight.data.normal_(0, 0.01)
#             if each_module.bias is not None:
#                 each_module.bias.data.zero_()

if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    __C.CONST = edict()
    cfg.CONST.N_VIEWS_RENDERING = 5
    model = paddle.Model(Res_Gru_Net(cfg))
    model.summary((16, 5, 3, 127, 127))