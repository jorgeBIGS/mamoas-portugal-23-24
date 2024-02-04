import os
import shutil
from inferencer import infere
from mmdetection.configs.mamoas_detection import *
from trainer import train



shutil.rmtree(TEMPORAL, ignore_errors=True)

os.makedirs(TEMPORAL, exist_ok=True)

os.makedirs(SHAPES_OUTPUT, exist_ok=True)
#['cascade_rcnn', 'faster_rcnn', 'retinanet', 'rpn']
models_l1 = ['cascade_rcnn', 'faster_rcnn', 'retinanet', 'rpn']
thresholds_min_l1 = [0]*4
thresholds_max_l1 = [1]*4
iou_threshold_l1 = [1]*4

models_l2 = ['cascade_rcnn', 'faster_rcnn', 'retinanet', 'rpn']
thresholds_min_l2 = [0]*4
thresholds_max_l2 = [1]*4
iou_threshold_l2 = [1]*4

models_l3 = ['cascade_rcnn', 'faster_rcnn', 'retinanet', 'rpn']
thresholds_min_l3 = [0]*4
thresholds_max_l3 = [1]*4
iou_threshold_l3 = [1]*4

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
        try:
        # Train model
            if  INCLUDE_TRAIN:
                train(config_file = MODEL_CONFIG_ROOT + model + '.py', work_dir_fold=MODEL_PATH + model,training_root=TRAINING_DATA_ROOT, validation_root=VAL_DATA_ROOT) 

            # Inference 
            infere(model, MODEL_PATH, MODEL_CONFIG_ROOT, TEST_IMAGE, threshold_min=thresholds_min_l1[i], threshold_max=thresholds_max_l1[i], iou_threshold_min=iou[i])
        except Exception as e:
            print('Error en ', model)
            print(e)
            print(25*'-----------------\n')