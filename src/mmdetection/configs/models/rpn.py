_base_ = [
    '../_base_/models/rpn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/runtime/default_runtime.py',
    '../mamoas_detection.py'
]
