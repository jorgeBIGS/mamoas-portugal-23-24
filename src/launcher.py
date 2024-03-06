import os

from data_generator import generate_dl_dataset, get_buffered_training
from trainer_eval import train_eval
from trainer_inferencer import apply_models, train_models
from auxiliar.mmdetection.configs.mamoas_detection import *

def main()->None:
    os.makedirs(SHAPES_OUTPUT_ROOT, exist_ok=True)
    training = get_buffered_training(TRUE_DATA,
                             BUFFER)
    
    generate_dl_dataset(IMAGE,  
                             training,
                             SIZE, 
                             SIZE/100,
                             OVERLAP, 
                             INCLUDE_ALL_IMAGES, 
                             LEAVE_ONE_OUT_BOOL, 
                             COMPLETE_BBOX_OVERLAP,
                             LENIENT_BBOX_OVERLAP_PERCENTAGE,
                             OUTPUT_DATA_ROOT, 
                             OUTPUT_SHAPE)
    
    #config_file=MODEL_CONFIG_ROOT + MODELS[0] + '.py'
    #work_dir = MODEL_PATH + MODELS[0] 
    #train_eval(config_file, work_dir, TRAINING_DATA_ROOT, TRAINING_DATA_ROOT, TRAINING_DATA_ROOT, SCORES)
    
    train_models(MODELS,OUTPUT_DATA_ROOT, OUTPUT_DATA_ROOT, MODEL_CONFIG_ROOT, MODEL_PATH)
    apply_models(MODELS, MODEL_CONFIG_ROOT, MODEL_PATH, TEST_IMAGE, TEMPORAL, SIZE, OVERLAP, SHAPES_OUTPUT_ROOT)


if __name__ == '__main__':
    main()