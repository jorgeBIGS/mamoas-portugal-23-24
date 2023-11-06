_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py',
    'mamoas_detection.py'
]


model = dict(
    # backbone=dict(
    #     type='ResNeXt',
    #     depth=101,
    #     groups=64,
    #     base_width=4,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], # Default:8
            ratios=[1.0],
            # base_sizes=[3,4,4,4,4],
            strides=[4, 8, 16, 32, 64]),
    ),
    ## Prueba
    # rpn_head=dict(
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[1], # Default:8
    #         ratios=[1.0],
    #         base_sizes=[30,32,128,256,512],
    #         strides=[4, 8, 16, 32, 64]),
    # ),
    
    roi_head=dict(
        bbox_roi_extractor=dict(
            # roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            # finest_scale=16,
        ),
        bbox_head=dict(
            # roi_feat_size=14,
            num_classes=1)
        ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05, # Default:0.05
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    )
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))

# We can use the pre-trained Faster-RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# Enable automatic-mixed precision to reduce computational cost
# optim_wrapper = dict(type='AmpOptimWrapper')

# Other things to adjust:
# Input resize (maybe multi-scale training/testing)
# Data preprocessing - normalization
# Anchors size (https://github.com/open-mmlab/mmdetection/issues/3669)
# Score threshold
# Use pre-trained from COCO? Seems to have slighlty better performance if weights are not loaded.
# Filter bboxes in train/test. Require min size of bbox