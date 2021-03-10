# -*- coding: utf-8 -*-

import paddle
import os
from datetime import datetime as dt


def var_or_cuda(x):
    return x.cuda()

# def init_weights(m):
#     if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
#         torch.nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)
#     elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d: #paddle有报错
#         torch.nn.init.constant_(m.weight, 1)
#         torch.nn.init.constant_(m.bias, 0)
#     elif type(m) == torch.nn.Linear:
#         torch.nn.init.normal_(m.weight, 0, 0.01)
#         torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, res_gru_net, res_gru_net_solver, best_iou, best_epoch):
    print('[INFO] %s Saving %s checkpoint to %s ...' % (dt.now(), epoch_idx, file_path))
    print('[INFO] best_epoch %s best_iou %s ...' % (best_epoch, best_iou.numpy ( ) ))
    paddle.save(res_gru_net.state_dict(), os.path.join(file_path, "res_gru_net.pdparams"))
    paddle.save(res_gru_net_solver.state_dict(), os.path.join(file_path, "res_gru_net_solver.pdopt"))
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
