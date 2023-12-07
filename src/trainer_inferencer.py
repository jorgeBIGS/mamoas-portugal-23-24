import os
import shutil
from inferencer import infere
from mmdetection.configs.mamoas_detection import *
from trainer import train

ONLY_INFERENCE = True

shutil.rmtree(TEMPORAL, ignore_errors=True)

os.makedirs(TEMPORAL, exist_ok=True)

os.makedirs(SHAPES_OUTPUT, exist_ok=True)
#['cascade_rcnn', 'faster_rcnn', 'retinanet', 'rpn']
models_l1 = ['retinanet']
thresholds_min_l1 = [0.90661181]
thresholds_max_l1 = [2.20171123]
iou_threshold_l1 = [0.5502254]

models_l2 = ['retinanet']
thresholds_min_l2 = [0.66433658]
thresholds_max_l2 = [0.86952232]
iou_threshold_l2 = [0.15279332]

models_l3 = ['retinanet']
thresholds_min_l3 = [0.52526017]
thresholds_max_l3 = [0.57069809]
iou_threshold_l3 = [0.72528968]

if LEVEL == 'L1':
    models = models_l1
    thresholds_min = thresholds_min_l1
    thresholds_max = thresholds_max_l1
    iou = iou_threshold_l1
elif LEVEL == 'L2':
    models = models_l2
    thresholds_min = thresholds_min_l2
    thresholds_max = thresholds_max_l2
    iou = iou_threshold_l2
else:
    models = models_l3
    thresholds_min = thresholds_min_l3
    thresholds_max = thresholds_max_l3
    iou = iou_threshold_l3

for i, model in enumerate(models):
    if thresholds_min[i] <= thresholds_max[i] and thresholds_min[i] <= 1:
        try:
            # Train model
            if not ONLY_INFERENCE:
                train(config_file = MODEL_CONFIG_ROOT + model + '.py', work_dir_fold=MODEL_PATH + model,training_root=TRAINING_DATA_ROOT, validation_root=VAL_DATA_ROOT) 

            # Inference 
            infere(model, MODEL_PATH, MODEL_CONFIG_ROOT, TEST_IMAGE, thresholds_min[i], thresholds_max[i], iou_threshold_min=iou[i])
        except Exception as e:
            print('Error en ', model)
            print(e)
            print(25*'-----------------\n')