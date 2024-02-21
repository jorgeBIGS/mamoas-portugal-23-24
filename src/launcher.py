import os

from data_generator import generate_training_dataset
from trainer_inferencer import apply_models, train_models
from auxiliar.mmdetection.configs.mamoas_detection import *

def main()->None:
    os.makedirs(SHAPES_OUTPUT, exist_ok=True)
    '''generate_training_dataset(IMAGE,  
                             TRUE_DATA,
                             BUFFER,
                             SIZE, 
                             SIZE/100,
                             OVERLAP, 
                             INCLUDE_ALL_IMAGES, 
                             LEAVE_ONE_OUT_BOOL, 
                             COMPLETE_BBOX_OVERLAP,
                             LENIENT_BBOX_OVERLAP_PERCENTAGE,
                             OUTPUT_DATA_ROOT, 
                             OUTPUT_SHAPE)'''
    
    train_models(MODELS,OUTPUT_DATA_ROOT, OUTPUT_DATA_ROOT, MODEL_CONFIG_ROOT, MODEL_PATH)
    #apply_models(MODELS, MODEL_CONFIG_ROOT, MODEL_PATH, TEST_IMAGE, TEMPORAL, SIZE, OVERLAP, SHAPES_OUTPUT)

if __name__ == '__main__':
    main()