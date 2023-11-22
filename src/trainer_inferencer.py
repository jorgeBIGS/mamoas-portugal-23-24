import subprocess

models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'fcos', 'retinanet', 'rpn', 'ssd', 'tood', 'yolo']
levels = ['L1', 'L2', 'L3']

for model in models:
    for level in levels:  
        # Train model
        subprocess.run(["python", "src/trainer.py", level , model])

        # Inference 
        subprocess.run(["python", "src/inferencer.py", level, model])        

