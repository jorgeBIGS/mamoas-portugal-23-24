from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.runner import Runner

TRAINING_DATA_ROOT = 'data/coco'
MODEL = "retinanet"

cfg = Config.fromfile('mmdetection/configs/mamoas/{MODEL}.py')
# Modify dataset classes and color
cfg.metainfo = {
    'CLASSES': ('mamoa', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

# Modify dataset type and path
cfg.data_root = TRAINING_DATA_ROOT

cfg.train_dataloader.dataset.ann_file = 'annotations/all.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'data/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'annotations/all.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'data/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader


# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/annotations/all.json'
cfg.test_evaluator = cfg.val_evaluator

# Modify num classes of the model in box head and mask head
#cfg.model.roi_head.bbox_head.num_classes = 1
#cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
#cfg.load_from = 'mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = 'src/detection'


# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 100
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 100

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optim_wrapper.optimizer.lr = 0.02 / 8
cfg.default_hooks.logger.interval = 10


# Set seed thus the results are more reproducible
# cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
#cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})


# build the runner from config
runner = Runner.from_cfg(cfg)
# start training
runner.train()