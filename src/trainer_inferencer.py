import os
import shutil
from inferencer import infere
from mmdetection.configs.mamoas_detection import *
from trainer import train


shutil.rmtree(TEMPORAL, ignore_errors=True)

os.makedirs(TEMPORAL, exist_ok=True)

os.makedirs(SHAPES_OUTPUT, exist_ok=True)
#['cascade_rcnn', 'faster_rcnn', 'retinanet']
models = ['faster_rcnn']
thresholds_min = [0.5]*3
thresholds_max = [1]*3
iou_threshold = [0.25]*3

for i, model in enumerate(models):
        try:
        # Train model
            if  INCLUDE_TRAIN:
                train(config_file = MODEL_CONFIG_ROOT + model + '.py', work_dir_fold=MODEL_PATH + model,training_root=OUTPUT_DATA_ROOT, validation_root=OUTPUT_DATA_ROOT) 

            # Inference 
            infere(model, MODEL_PATH, MODEL_CONFIG_ROOT, TEST_IMAGE, threshold_min=thresholds_min[i], threshold_max=thresholds_max[i], iou_threshold_min=iou_threshold[i])
        except Exception as e:
            print('Error en ', model)
            print(e)
            print(25*'-----------------\n')