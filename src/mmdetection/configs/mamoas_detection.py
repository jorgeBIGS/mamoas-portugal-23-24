ORIGINALES = 'data/original'
SHAPES_OUTPUT = 'data/shapes'

# Parámetros generación datasets
IMAGE = ORIGINALES + '/COMB-Laboreiro.tif'
TRAINING_SHAPE = ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'
TRUE_DATA = ORIGINALES + '/Mamoas-Laboreiro.shp'
SIZE = 200
OVERLAP = [0, 100]
INCLUDE_ALL_IMAGES = False
LEAVE_ONE_OUT_BOOL = False
COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5
OUTPUT_DATA_ROOT= 'data/mamoas-laboreiro/'
OUTPUT_SHAPE = SHAPES_OUTPUT + '/Mamoas-Laboreiro.shp'



#parámetros de preprocesamiento
#BUFFER_SIZES = [2.5, 5, 7.5]
#RES_MIN = BUFFER_SIZES[0]
#PERCENTILE = 0.5

#parámetros de training

#parámetros de optimización
#NUM_GENERATIONS=50
#NUM_INDIVIDUALS=100
#NUM_PARENT_MATING = 2
#ELITISM = 2
#MUTATION_PERCENT = 80
#NUM_THREADS = 15

#parámetros de training/inference
MODEL_CONFIG_ROOT = "src/mmdetection/configs/models/"
MODEL_PATH = 'data/model/'
INCLUDE_TRAIN = True
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
TEMPORAL = 'data/tmp'

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
        data_root=OUTPUT_DATA_ROOT,
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
        data_root=OUTPUT_DATA_ROOT,
        metainfo=metainfo,
        ann_file='annotations/all.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=OUTPUT_DATA_ROOT + 'annotations/all.json',
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


