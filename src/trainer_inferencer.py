import os
import shutil
from inferencer import infere
from mmdetection.configs.mamoas.mamoas_detection import *
from trainer import train

shutil.rmtree(TEMPORAL, ignore_errors=True)

os.makedirs(TEMPORAL, exist_ok=True)

os.makedirs(SHAPES_OUTPUT, exist_ok=True)

models_l1 = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
thresholds_min_l1 = [3.89949643,0.9481472, 2.39667152,0.90608631,3.29919663, -3.30883177,-0.47856092]
thresholds_max_l1 = [5.33753824,-3.51117828,4.31697631,0.94211238,3.78307203,-3.07835448,-1.15154828]

models_l2 = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
thresholds_min_l2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
thresholds_max_l2 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]

models_l3 = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
thresholds_min_l3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
thresholds_max_l3 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]

#models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
#thresholds_l1 = [0.93586152, 0.96225211, 0.92739374, 0.80668515, 2.88025115, 0.89356221, 0.6531718]

#models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
#thresholds_l1 = [0.94143573 ,0.96264526, 0.94799123 ,0.808563 ,  2.76978491 ,0.91544499 ,0.64645162]

#models_l1 = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
#thresholds_l1 = [0.93407015, 0.95631426, 0.92838168, 0.81619744, 9.32740101, 0.89433167, 0.64730994] #Fitness: 503
  
#models_l2 = ['retinanet']
#thresholds_min_l2 = [0.43588853] #Fitness: 1248.0
#thresholds_max_l2 = [1.0]

#models_l3 = ['cascade_rcnn', 'faster_rcnn', 'retinanet']
#thresholds_min_l3 = [0.93035984,  0.90768589, 0.39439886  ]
#thresholds_max_l3 = [1.0, 1.0, 1.0]

if LEVEL == 'L1':
    models = models_l1
    thresholds_min = thresholds_min_l1
    thresholds_max = thresholds_max_l1
elif LEVEL == 'L2':
    models = models_l2
    thresholds_min = thresholds_min_l2
    thresholds_max = thresholds_max_l2
else:
    models = models_l3
    thresholds_min = thresholds_min_l3
    thresholds_max = thresholds_max_l3

for i, model in enumerate(models):
    if thresholds_min[i] <= thresholds_max[i] and thresholds_min[i] <= 1:
        try:
        # Train model
            train(config_file = MODEL_CONFIG_ROOT + model + '.py', work_dir_fold=MODEL_PATH + model, data_root=TRAINING_DATA_ROOT) 

        # Inference 
            infere(model, MODEL_PATH, MODEL_CONFIG_ROOT, TEST_IMAGE, thresholds_min[i], thresholds_max[i])
        except:
            print('Error en ', model)
            print(25*'-----------------\n')