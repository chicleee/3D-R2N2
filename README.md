# 3D-R2N2

![](https://ai-studio-static-online.cdn.bcebos.com/4c23295123e8467f99c06f6fdb9d32ccdd26e805402f4a5592ac9441ed448953)

This is a PaddlePaddle2.0 implementation of the paper [《3D-R2N2: A Unified Approach for Single andMulti-view 3D Object Reconstruction》ECCV 2016.](http://d-r2n2.stanford.edu/). by Choy et al. Given one or multiple views of an object, the network generates voxelized ( a voxel is the 3D equivalent of a pixel) reconstruction of the object in 3D.

See the [official repo](http://https://github.com/chrischoy/3D-R2N2) in Theano, as well as overview of the method.

[AI Studio Notebook.](https://aistudio.baidu.com/aistudio/projectdetail/1631256) 

For now, only the residual GRU-based architecture with neighboring recurrent unit connection is implemented. It is called 
**Res3D-GRU-3** in the paper.

**differences**

1. The loss function is **BCE** (the mean value of the voxel-wise binary cross entropies between the reconstructed object and the ground truth. ) instead of **CE** in the original paper.

1. For a fair comparison， the same data augment strategy as [Pix2Vox](http://rxiv.org/abs/1901.11153) was used in this experiment.

# Dataset

Use the same dataset as mentioned in the official repo.

--ShapeNet rendered images [http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

--ShapeNet voxelized models [http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

--Pix3D images & voxelized models: [http://pix3d.csail.mit.edu/data/pix3d.zip](http://pix3d.csail.mit.edu/data/pix3d.zip)

The dataset is already mounted in this notebook.


```python
!unzip -oq data/data67155/dataset.zip
```

# Install Python Denpendencies


```python
%cd work/3D-R2N2/
!pip install -r requirements.txt
```

# Get Started

Update Settings in work/3D-R2N2/config.py


```python
# To train 3D-R2N2:
%cd work/3D-R2N2/
!python3 runner.py
```

[download best checkpoint](https://aistudio.baidu.com/aistudio/datasetdetail/73641) 


```python
# Test
!python3 runner.py --test --weights=/home/aistudio/work/3D-R2N2/output/checkpoints/2021-03-09T16:19:52.385245/best-ckpt
```

    Use config:
    {'CONST': {'BATCH_SIZE': 30,
               'CROP_IMG_H': 96,
               'CROP_IMG_W': 96,
               'DEVICE': '0',
               'IMG_H': 127,
               'IMG_W': 127,
               'INFO_BATCH': 100,
               'N_VIEWS_RENDERING': 5,
               'N_VOX': 32,
               'RNG_SEED': 0,
               'WEIGHTS': '/home/aistudio/work/3D-R2N2/output/checkpoints/2021-03-09T16:19:52.385245/best-ckpt'},
     'DATASET': {'MEAN': [0.5, 0.5, 0.5],
                 'STD': [0.5, 0.5, 0.5],
                 'TEST_DATASET': 'ShapeNet',
                 'TRAIN_DATASET': 'ShapeNet'},
     'DATASETS': {'SHAPENET': {'RENDERING_PATH': '/home/aistudio/dataset/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png',
                               'TAXONOMY_FILE_PATH': './datasets/ShapeNet.json',
                               'VOXEL_PATH': '/home/aistudio/dataset/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'}},
     'DIR': {'OUT_PATH': './output'},
     'NETWORK': {'LEAKY_VALUE': 0.2, 'TCONV_USE_BIAS': False, 'USE_MERGER': True},
     'TEST': {'RANDOM_BG_COLOR_RANGE': [[240, 240], [240, 240], [240, 240]],
              'VOXEL_THRESH': [0.2, 0.3, 0.4, 0.5]},
     'TRAIN': {'BETAS': [0.9, 0.999],
               'BRIGHTNESS': 0.4,
               'CONTRAST': 0.4,
               'GAMMA': 0.5,
               'MOMENTUM': 0.9,
               'NOISE_STD': 0.1,
               'NUM_EPOCHES': 60,
               'NUM_WORKER': 4,
               'POLICY': 'adam',
               'RANDOM_BG_COLOR_RANGE': [[225, 255], [225, 255], [225, 255]],
               'RESUME_TRAIN': False,
               'RES_GRU_NET_LEARNING_RATE': 0.0001,
               'RES_GRU_NET_LR_MILESTONES': [45],
               'SATURATION': 0.4,
               'SAVE_FREQ': 10,
               'UPDATE_N_VIEWS_RENDERING': False}}
    [INFO] 2021-03-10 15:27:56.780385 Collecting files of Taxonomy[ID=02691156, Name=aeroplane]
    [INFO] 2021-03-10 15:27:56.992252 Collecting files of Taxonomy[ID=02828884, Name=bench]
    [INFO] 2021-03-10 15:27:57.052724 Collecting files of Taxonomy[ID=02933112, Name=cabinet]
    [INFO] 2021-03-10 15:27:57.111380 Collecting files of Taxonomy[ID=02958343, Name=car]
    [INFO] 2021-03-10 15:27:57.415542 Collecting files of Taxonomy[ID=03001627, Name=chair]
    [INFO] 2021-03-10 15:27:57.631712 Collecting files of Taxonomy[ID=03211117, Name=display]
    [INFO] 2021-03-10 15:27:57.668383 Collecting files of Taxonomy[ID=03636649, Name=lamp]
    [INFO] 2021-03-10 15:27:57.744364 Collecting files of Taxonomy[ID=03691459, Name=speaker]
    [INFO] 2021-03-10 15:27:57.818193 Collecting files of Taxonomy[ID=04090263, Name=rifle]
    [INFO] 2021-03-10 15:27:57.927521 Collecting files of Taxonomy[ID=04256520, Name=sofa]
    [INFO] 2021-03-10 15:27:58.070311 Collecting files of Taxonomy[ID=04379243, Name=table]
    [INFO] 2021-03-10 15:27:58.444920 Collecting files of Taxonomy[ID=04401088, Name=telephone]
    [INFO] 2021-03-10 15:27:58.492240 Collecting files of Taxonomy[ID=04530566, Name=watercraft]
    [INFO] 2021-03-10 15:27:58.569851 Complete collecting files of the dataset: ShapeNet. Total files: 8770.
    [INFO] Collected files of testet
    W0310 15:27:58.595484 16582 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0310 15:27:58.599697 16582 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    [INFO] 2021-03-10 15:28:02.028381 Loading weights from /home/aistudio/work/3D-R2N2/output/checkpoints/2021-03-09T16:19:52.385245/best-ckpt ...
    ============================ TEST RESULTS ============================
    Taxonomy	#Sample	Baseline	t=0.20	t=0.30	t=0.40	t=0.50	
    aeroplane	810	0.5610		0.6282	0.6444	0.6404	0.6215	
    bench   	364	0.5270		0.5958	0.6050	0.5934	0.5629	
    cabinet 	315	0.7720		0.7971	0.8017	0.7997	0.7907	
    car     	1501	0.8360		0.8515	0.8607	0.8619	0.8569	
    chair   	1357	0.5500		0.5974	0.6057	0.5983	0.5781	
    display 	220	0.5650		0.5905	0.5989	0.5925	0.5714	
    lamp    	465	0.4210		0.4744	0.4601	0.4342	0.4001	
    speaker 	325	0.7170		0.7405	0.7408	0.7331	0.7185	
    rifle   	475	0.6000		0.6231	0.6295	0.6163	0.5874	
    sofa    	635	0.7060		0.7393	0.7480	0.7449	0.7304	
    table   	1703	0.5800		0.6295	0.6321	0.6209	0.6009	
    telephone	211	0.7540		0.7799	0.7942	0.7989	0.7965	
    watercraft	389	0.6100		0.6292	0.6417	0.6363	0.6154	
    Overall 				0.6731	0.6799	0.6730	0.6553	
    
    [INFO/MainProcess] process shutting down


# Result

**Results in the paper:**

![](https://ai-studio-static-online.cdn.bcebos.com/8c54936eeda04001b1aa65ac1f6eee0bfb45b989da624214b097ab0cd0b330bd)

**Results in this experiment:**


**5 Views, Valid BCE = 0.7134, Valid IoU = 0.68196**, 

**On the test set, when t=0.30， IoU=0.6799**



# references

> https://github.com/heromanba/3D-R2N2-PyTorch

> https://github.com/chrischoy/3D-R2N2

> https://github.com/hzxie/Pix2Vox

# License

This project is open sourced under MIT license.
