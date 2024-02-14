from data_generator import generate_training_dataset
from mmdetection.configs.mamoas_detection import *

def main()->None:
    generate_training_dataset(IMAGE, 
                             TRAINING_SHAPE, 
                             TRUE_DATA,
                             SIZE, 
                             SIZE/100,
                             OVERLAP, 
                             INCLUDE_ALL_IMAGES, 
                             LEAVE_ONE_OUT_BOOL, 
                             COMPLETE_BBOX_OVERLAP,
                             LENIENT_BBOX_OVERLAP_PERCENTAGE,
                             OUTPUT_DATA_ROOT, 
                             OUTPUT_SHAPE)

if __name__ == '__main__':
    main()