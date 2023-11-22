import subprocess

from mmdetection.configs.mamoas.mamoas_detection import *

models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'fcos', 'retinanet', 'rpn', 'ssd', 'tood', 'yolo']
levels = ['L1', 'L2', 'L3']

for model in models:
    MODEL = model
    for level in levels:
        LEVEL = level  
        # Train model
        subprocess.run(["python", "src/trainer.py"])

        # Inference 
        subprocess.run(["python", "src/inferencer.py"])        

