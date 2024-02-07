_base_ =[
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/runtime/default_runtime.py',
    '../mamoas_detection.py'
] 

model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg = dict(max_epochs=24, val_interval=1),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5, # Default:0.05
            nms=dict(type='nms', iou_threshold=0.25),
            max_per_img=5))
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))