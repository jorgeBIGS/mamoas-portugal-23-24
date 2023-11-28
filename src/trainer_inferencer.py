from inferencer import infere
from mmdetection.configs.mamoas.mamoas_detection import *
from trainer import train

#models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
#thresholds_l1 = [0.93586152, 0.96225211, 0.92739374, 0.80668515, 2.88025115, 0.89356221, 0.6531718]

#models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
#thresholds_l1 = [0.94143573 ,0.96264526, 0.94799123 ,0.808563 ,  2.76978491 ,0.91544499 ,0.64645162]

models_l1 = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'retinanet', 'rpn', 'ssd', 'yolo']
thresholds_l1 = [0.93407015, 0.95631426, 0.92838168, 0.81619744, 9.32740101, 0.89433167, 0.64730994] #Fitness: 503

models_l2 = ['retinanet']
thresholds_l2 = [0.43588853] #Fitness: 1248.0

models_l3 = ['cascade_rcnn', 'faster_rcnn', 'retinanet']
thresholds_l3 = [0.93035984,  0.90768589, 0.39439886  ]

models = models_l3
thresholds = thresholds_l3

for i, model in enumerate(models):
    # Train model
    #train(config_file = MODEL_CONFIG_ROOT + model + '.py', work_dir_fold=MODEL_PATH + model, data_root=TRAINING_DATA_ROOT) 

    # Inference 
    infere(model, MODEL_PATH, MODEL_CONFIG_ROOT, TEST_IMAGE, thresholds[i])
   
