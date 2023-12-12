_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/runtime/default_runtime.py',
    '../mamoas_detection.py'
]

model = dict(
    bbox_head=dict(num_classes=1),
    train_cfg = dict(max_epochs=24, val_interval=1),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05, # Default:0.05
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))