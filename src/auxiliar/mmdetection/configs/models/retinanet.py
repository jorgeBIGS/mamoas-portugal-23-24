

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/runtime/default_runtime.py',
    '../mamoas_detection.py'
]
NUM_CLASSES = 1

model = dict(
    bbox_head=dict(
        type='RetinaHead',
        num_classes=NUM_CLASSES,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    train_cfg = dict(max_epochs=24, val_interval=1),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05, # Default:0.05
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=50))
))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))