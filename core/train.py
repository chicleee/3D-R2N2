# -*- coding: utf-8 -*-

import os
import random
import paddle
from visualdl import LogWriter

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from time import time
from core.test import test_net
from models.network import Res_Gru_Net

def train_net(cfg):
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = paddle.io.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    #num_workers=0  , # cfg.TRAIN.NUM_WORKER>0时报错，因为dev/shm/太小  https://blog.csdn.net/ctypyb2002/article/details/107914643
                                                    #pin_memory=True,
                                                    use_shared_memory=False,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = paddle.io.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  #num_workers=1,
                                                  #pin_memory=True,
                                                  shuffle=False)

    # Set up networks # paddle.Model prepare fit save
    res_gru_net = Res_Gru_Net(cfg)

    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(res_gru_net)))

    # Set up learning rate scheduler to decay learning rates dynamically
    res_gru_net_lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.RES_GRU_NET_LEARNING_RATE,
                                                               milestones=cfg.TRAIN.RES_GRU_NET_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA, verbose=True)
    # Set up solver
    # if cfg.TRAIN.POLICY == 'adam':
    res_gru_net_solver = paddle.optimizer.Adam(learning_rate=res_gru_net_lr_scheduler, parameters=res_gru_net.parameters())

    # Set up loss functions
    bce_loss = paddle.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        # load
        res_gru_net_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "res_gru_net.pdparams"))
        res_gru_net_solver_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "res_gru_net_solver.pdopt"))
        res_gru_net.set_state_dict(res_gru_net_state_dict)
        res_gru_net_solver.set_state_dict(res_gru_net_solver_state_dict)

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    # train_writer = SummaryWriter()
    # val_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    train_writer=LogWriter(os.path.join(log_dir, 'train'))
    val_writer=LogWriter(os.path.join(log_dir, 'val'))
  
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        res_gru_net_losses = utils.network_utils.AverageMeter()

        # # switch models to training mode
        res_gru_net.train()
        batch_end_time = time()
        n_batches = len(train_data_loader)
        
        for batch_idx, (rendering_images, ground_truth_volumes) in enumerate(train_data_loader()):
            # if batch_idx>1:
            #     exit()
            # Measure data time
            data_time.update(time() - batch_end_time)
            rendering_images = rendering_images.cuda()
            ground_truth_volumes = ground_truth_volumes.cuda()
            # print(rendering_images.shape)
            # print(ground_truth_volumes.shape)
            # [64, 5, 3, 224, 224]
            # [64, 32, 32, 32]
            # print("ground_truth_volumes", ground_truth_volumes)

            # Train the res_gru_net
            generated_volumes = res_gru_net(rendering_images)
            # print("generated_volumes", generated_volumes)
            res_gru_net_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            res_gru_net_loss.backward()
            res_gru_net_solver.step()
            # Gradient decent
            res_gru_net_solver.clear_grad ()

            # Append loss to average metrics
            res_gru_net_losses.update(res_gru_net_loss)
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar(tag='Res_Gru_Net/BatchLoss', step=n_itr, value=res_gru_net_loss)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            n_batches = len(train_data_loader)
            if (batch_idx % int(cfg.CONST.INFO_BATCH )) == 0:
                print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f' %
                    (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                    data_time.val, res_gru_net_loss))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar(tag='Res_Gru_Net/EpochLoss', step=epoch_idx + 1, value=res_gru_net_losses.avg)


        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, res_gru_net_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, res_gru_net)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d' % (epoch_idx + 1)),
                                                 epoch_idx + 1, res_gru_net, res_gru_net_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt'), epoch_idx + 1, 
            res_gru_net, res_gru_net_solver, best_iou, best_epoch)
