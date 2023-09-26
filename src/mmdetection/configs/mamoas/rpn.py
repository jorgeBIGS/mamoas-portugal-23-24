_base_ = [
    '../_base_/models/rpn_r50_fpn.py',
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
            strides=[4, 8, 16, 32, 64]
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0
        )
    )
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))
