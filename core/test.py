# -*- coding: utf-8 -*-
#

import json
import numpy as np
import os
import paddle

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.network import Res_Gru_Net



def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             res_gru_net=None):
   
    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = paddle.io.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                    #    num_workers=1,
                                                       shuffle=False)
        mode = 'test'
    else:
        mode = 'val'

    
    # paddle.io.Dataset not support 'str' input
    dataset_taxonomy = None
    rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
    volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH

    # Load all taxonomies of the dataset
    with open('./datasets/ShapeNet.json', encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
        # print("[INFO]TEST-- open TAXONOMY_FILE_PATH succeess")

    all_test_taxonomy_id_and_sample_name = []
    # Load data for each category
    for taxonomy in dataset_taxonomy:
        taxonomy_folder_name = taxonomy['taxonomy_id']
        # print('[INFO] %set -- Collecting files of Taxonomy[ID=%s, Name=%s]' %
        #         (mode, taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
        samples = taxonomy[mode]
        for sample in samples:
            all_test_taxonomy_id_and_sample_name.append([taxonomy_folder_name, sample])
    # print(len(all_test_taxonomy_id_and_sample_name))
    # print(all_test_taxonomy_id_and_sample_name)
    print('[INFO] Collected files of %set' % (mode))   
    # Set up networks
    if res_gru_net is None:
        res_gru_net = Res_Gru_Net(cfg)

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        res_gru_net_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "res_gru_net.pdparams"))
        res_gru_net.set_state_dict(res_gru_net_state_dict)

    # Set up loss functions
    bce_loss = paddle.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    res_gru_net_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    res_gru_net.eval()

    for sample_idx, (rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = all_test_taxonomy_id_and_sample_name[sample_idx][0]
        sample_name = all_test_taxonomy_id_and_sample_name[sample_idx][1]
        # print("all_test_taxonomy_id_and_sample_name")
        # print(taxonomy_id)
        # print(sample_name)

        with paddle.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the res_gru_net, decoder and merger
            generated_volume = res_gru_net(rendering_images)

            res_gru_net_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            # Append loss and accuracy to average metrics
            res_gru_net_losses.update(res_gru_net_loss)

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = paddle.greater_equal(generated_volume, paddle.to_tensor(th)).astype("float32")
                intersection = paddle.sum(paddle.multiply(_volume, ground_truth_volume))
                union = paddle.sum(paddle.greater_equal(paddle.add(_volume, ground_truth_volume).astype("float32"), paddle.to_tensor(1., dtype='float32')).astype("float32"))
                sample_iou.append((intersection / union))

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx < 1:
                img_dir = output_dir % 'images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'Reconstructed'),
                                                                              epoch_idx)
                test_writer.add_image(tag='Reconstructed', img=rendering_views, step=epoch_idx)
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'GroundTruth'),
                                                                              epoch_idx)
                test_writer.add_image(tag='GroundTruth', img=rendering_views, step=epoch_idx)

            # # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f IoU = %s' %
                  (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, res_gru_net_loss,
                   ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar(tag='Res_Gru_Net/EpochLoss', value=res_gru_net_losses.avg, step=epoch_idx)
        test_writer.add_scalar(tag='Res_Gru_Net/IoU', value=max_iou, step=epoch_idx)

    return max_iou
