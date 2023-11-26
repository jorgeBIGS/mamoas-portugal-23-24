import subprocess

#models = ['cascade_rcnn', 'dynamic_rcnn', 'faster_rcnn', 'fcos', 'retinanet', 'rpn', 'ssd', 'tood', 'yolo']
models = ['rpn']
levels = ['L2']

for model in models:
    for level in levels:  
        # Train model
        subprocess.run(["python", "src/trainer.py", level , model])

        # Inference 
        subprocess.run(["python", "src/inferencer.py", level, model])        

