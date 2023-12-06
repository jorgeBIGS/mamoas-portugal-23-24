LEVEL = 'L1'

#parámetros de preprocessing
SIZE_L1 = 200
OVERLAP_L1 = [0, SIZE_L1//2]
MODEL_PATH_L1 = 'data/model_l1/'

OUTPUT_DATA_ROOT_L1= 'data/mamoas-laboreiro_l1/'
DST_IMAGE_DIR_L1 = OUTPUT_DATA_ROOT_L1 + "tiles/"
DST_VALID_TILES_L1 = OUTPUT_DATA_ROOT_L1 + "valid_tiles/"
DST_DATA_ANNOTATION_L1 = OUTPUT_DATA_ROOT_L1 + "annotations/"
DST_DATA_LOO_CV_L1 = DST_DATA_ANNOTATION_L1 + "loo_cv/"
DST_DATA_IMAGES_L1 = OUTPUT_DATA_ROOT_L1 + "images/"

SIZE_L2 = 500
OVERLAP_L2 = [0, SIZE_L2//2]
MODEL_PATH_L2 = 'data/model_l2/'

OUTPUT_DATA_ROOT_L2 = 'data/mamoas-laboreiro_l2/'
DST_IMAGE_DIR_L2 = OUTPUT_DATA_ROOT_L2 + "tiles/"
DST_VALID_TILES_L2 = OUTPUT_DATA_ROOT_L2 + "valid_tiles/"
DST_DATA_ANNOTATION_L2 = OUTPUT_DATA_ROOT_L2 + "annotations/"
DST_DATA_LOO_CV_L2 = DST_DATA_ANNOTATION_L2 + "loo_cv/"
DST_DATA_IMAGES_L2 = OUTPUT_DATA_ROOT_L2 + "images/"

SIZE_L3 = 1000
OVERLAP_L3 = [0, SIZE_L3//2]
MODEL_PATH_L3 = 'data/model_l3/'

OUTPUT_DATA_ROOT_L3 = 'data/mamoas-laboreiro_l3/'
DST_IMAGE_DIR_L3 = OUTPUT_DATA_ROOT_L3 + "tiles/"
DST_VALID_TILES_L3 = OUTPUT_DATA_ROOT_L3 + "valid_tiles/"
DST_DATA_ANNOTATION_L3 = OUTPUT_DATA_ROOT_L3 + "annotations/"
DST_DATA_LOO_CV_L3 = DST_DATA_ANNOTATION_L3 + "loo_cv/"
DST_DATA_IMAGES_L3 = OUTPUT_DATA_ROOT_L3 + "images/"

RES_MIN = 5
PERCENTILE = 0.5
ORIGINALES = 'data/original'
TRUE_IMAGE=ORIGINALES + '/COMB-Laboreiro_2.tif'
TRUE_SHAPE=ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'

LEAVE_ONE_OUT_BOOL = False
INCLUDE_ALL_IMAGES = False

COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5

#parámetros de training
MODEL_CONFIG_ROOT = "src/mmdetection/configs/mamoas/"

if LEVEL == 'L1':
    MODEL_PATH = MODEL_PATH_L1
    TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_L1 + 'training/'
    VAL_DATA_ROOT = OUTPUT_DATA_ROOT_L1 + 'validation/'
    SIZE = SIZE_L1
    OVERLAP = OVERLAP_L1
elif LEVEL == 'L2':
    MODEL_PATH = MODEL_PATH_L2
    TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_L2 + 'training/'
    VAL_DATA_ROOT = OUTPUT_DATA_ROOT_L2 + 'validation/'
    SIZE = SIZE_L2
    OVERLAP = OVERLAP_L2
else:
    MODEL_PATH = MODEL_PATH_L3
    TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_L3 + 'training/'
    VAL_DATA_ROOT = OUTPUT_DATA_ROOT_L3 + 'validation/'
    SIZE = SIZE_L3
    OVERLAP = OVERLAP_L3

#parámetros de optimización
NUM_GENERATIONS=50
NUM_INDIVIDUALS=100
NUM_PARENT_MATING = 2
ELITISM = 2
MUTATION_PERCENT = 80
NUM_THREADS = 10

TRUE_DATA = 'data/original/Mamoas-Laboreiro.shp'
SHP_DIRECTORY = 'data/shapes/' + LEVEL

#parámetros de inference
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
TEMPORAL = 'data/tmp'
SHAPES_OUTPUT = 'data/shapes'


dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('mamoa', ),
    'palette': [
        (220, 20, 60),
    ]
}

backend_args = None

# img_scales = [(200, 200), (400, 400), (600, 600)]
img_scales = [(SIZE, SIZE)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomChoiceResize', scales=img_scales, keep_ratio=True),
    dict(type='Resize', scale=(SIZE, SIZE), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(SIZE,SIZE), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# tta_model = dict(
#     type='DetTTAModel',
#     tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))



# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='TestTimeAug',
#      transforms=[
#         [dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales],
#         [dict(type='RandomFlip', prob=1.),
#          dict(type='RandomFlip', prob=0.)],
#         [dict(type='LoadAnnotations', with_bbox=True)],
#         [dict(type='PackDetInputs',
#               meta_keys=('img_id', 'img_path', 'ori_shape',
#                          'img_shape', 'scale_factor', 'flip',
#                          'flip_direction'))]])
# ]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=TRAINING_DATA_ROOT,
        metainfo=metainfo,
        ann_file='annotations/all.json',
        data_prefix=dict(img='images/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=VAL_DATA_ROOT,
        metainfo=metainfo,
        ann_file='annotations/all.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=VAL_DATA_ROOT + 'annotations/all.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator


model = dict(
    bbox_head=dict(num_classes=1),
    train_cfg = dict(max_epochs=24, val_interval=1),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05, # Default:0.05
            nms=dict(type='nms', iou_threshold=0.95),
            max_per_img=100))
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)
]


default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=-1))