import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Import the module from the parent directory
from preprocessing import generate_coco_annotations, get_training
from mmdetection.configs.mamoas.mamoas_detection import TRAINING_SHAPE

training = get_training(TRAINING_SHAPE)

print(len(training), "[51 en Arcos, 86 en Laboreiro]")


generate_coco_annotations(['02128_.tif', '09607_.tif'], training, "aux.json")