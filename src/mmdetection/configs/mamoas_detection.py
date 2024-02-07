LEVEL = 'L2'

#parámetros de preprocessing
BUFFER_SIZES = [2.5, 5, 7.5]
RES_MIN = BUFFER_SIZES[0]
PERCENTILE = 0.5
ORIGINALES = 'data/original'
TRUE_IMAGE=ORIGINALES + '/COMB-Laboreiro.tif'
TRUE_SHAPE= 'data/training/COMB-LaboreiroL1-faster_rcnn.shp'#ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'
TRUE_DATA = 'data/original/Mamoas-Laboreiro.shp'

LEAVE_ONE_OUT_BOOL = False
INCLUDE_ALL_IMAGES = False

COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5


SIZE_L1 = 200
OVERLAP_L1 = [0, SIZE_L1//2]
MODEL_PATH_L1 = 'data/model_l1/'

OUTPUT_DATA_ROOT_L1= 'data/mamoas-laboreiro_l1/'
DST_IMAGE_DIR_L1 = OUTPUT_DATA_ROOT_L1 + "tiles/"
DST_VALID_TILES_L1 = OUTPUT_DATA_ROOT_L1 + "valid_tiles/"
DST_DATA_ANNOTATION_L1 = OUTPUT_DATA_ROOT_L1 + "annotations/"
DST_DATA_LOO_CV_L1 = DST_DATA_ANNOTATION_L1 + "loo_cv/"
DST_DATA_IMAGES_L1 = OUTPUT_DATA_ROOT_L1 + "images/"

SIZE_L2 = SIZE_L1
OVERLAP_L2 = OVERLAP_L1
MODEL_PATH_L2 = 'data/model_l2/'

OUTPUT_DATA_ROOT_L2 = 'data/mamoas-laboreiro_l2/'
DST_IMAGE_DIR_L2 = OUTPUT_DATA_ROOT_L2 + "tiles/"
DST_VALID_TILES_L2 = OUTPUT_DATA_ROOT_L2 + "valid_tiles/"
DST_DATA_ANNOTATION_L2 = OUTPUT_DATA_ROOT_L2 + "annotations/"
DST_DATA_LOO_CV_L2 = DST_DATA_ANNOTATION_L2 + "loo_cv/"
DST_DATA_IMAGES_L2 = OUTPUT_DATA_ROOT_L2 + "images/"


#parámetros de training
MODEL_CONFIG_ROOT = "src/mmdetection/configs/models/"

if LEVEL == 'L1':
    MODEL_PATH = MODEL_PATH_L1
    TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_L1
    VAL_DATA_ROOT = OUTPUT_DATA_ROOT_L1 
    SIZE = SIZE_L1
    OVERLAP = OVERLAP_L1
else:
    MODEL_PATH = MODEL_PATH_L2
    TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_L2
    VAL_DATA_ROOT = OUTPUT_DATA_ROOT_L2
    SIZE = SIZE_L2
    OVERLAP = OVERLAP_L2


#parámetros de optimización
NUM_GENERATIONS=50
NUM_INDIVIDUALS=100
NUM_PARENT_MATING = 2
ELITISM = 2
MUTATION_PERCENT = 80
NUM_THREADS = 15


SHP_DIRECTORY = 'data/shapes'

#parámetros de inference
INCLUDE_TRAIN = True
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
TEMPORAL = 'data/tmp'
SHAPES_OUTPUT = 'data/shapes'


dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('mamoa', 'not_mamoa'),
    'palette': [
        (220, 20, 60), (119, 11, 32)
    ]
}

backend_args = None

# img_scales = [(200, 200), (400, 400), (600, 600)]
img_scales = [(SIZE, SIZE)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True), #, with_masks=True
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


