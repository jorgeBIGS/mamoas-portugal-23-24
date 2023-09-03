_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py',
    'mamoas_detection.py'
]

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4], # Default:8
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1)
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