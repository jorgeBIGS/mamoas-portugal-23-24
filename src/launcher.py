import os
import itertools
from data_generator import difference, generate_dl_dataset, get_buffered_training
from trainer_eval import train_eval
from trainer_inferencer import apply_models, train_models
from auxiliar.mmdetection.configs.mamoas_detection import *
import geopandas as gpd

def train_contrastive_network(shape_file:str, training):
    gdf = gpd.read_file(shape_file)
    # self join on geodataframe to get all polygon intersections
    removing = gpd.sjoin(gdf, training, how='left', predicate='intersects')
    removing = removing.dissolve('index_right')

    geodata = gpd.overlay(gdf, removing, how='difference', keep_geom_type=False, make_valid=True)
    # Guarda el archivo shapefile
    geodata.to_file(shape_file.replace(".shp", "-no-train.shp"))

    generate_dl_dataset(IMAGE,  
                             gdf,
                             SIZE, 
                             SIZE/100,
                             OVERLAP, 
                             INCLUDE_ALL_IMAGES, 
                             LEAVE_ONE_OUT_BOOL, 
                             COMPLETE_BBOX_OVERLAP,
                             LENIENT_BBOX_OVERLAP_PERCENTAGE,
                             OUTPUT_DATA_ROOT_2, 
                             OUTPUT_SHAPE)

def main()->None:
    #os.makedirs(SHAPES_OUTPUT_ROOT, exist_ok=True)
    training = get_buffered_training(TRUE_DATA,
                             BUFFER)
    
    #generate_dl_dataset(IMAGE,  #
    #                         training,
    #                         SIZE, 
    #                         SIZE/100,
    #                         OVERLAP, 
    #                         INCLUDE_ALL_IMAGES, 
    #                         LEAVE_ONE_OUT_BOOL, 
    #                         COMPLETE_BBOX_OVERLAP,
    #                         LENIENT_BBOX_OVERLAP_PERCENTAGE,
    #                         OUTPUT_DATA_ROOT, 
    #                         OUTPUT_SHAPE)
    
    #config_file=MODEL_CONFIG_ROOT + MODELS[0] + '.py'
    #work_dir = MODEL_PATH + MODELS[0] 
    #train_eval(config_file, work_dir, TRAINING_DATA_ROOT, TRAINING_DATA_ROOT, TRAINING_DATA_ROOT, SCORES)
    
    #train_models(MODELS,OUTPUT_DATA_ROOT, OUTPUT_DATA_ROOT, MODEL_CONFIG_ROOT, MODEL_PATH)
    #apply_models(MODELS, MODEL_CONFIG_ROOT, MODEL_PATH, 
    #             TRAINING_IMAGE, TEMPORAL, SIZE, OVERLAP, SHAPES_OUTPUT_ROOT, 0.5, 1.0, 0.5)
    train_contrastive_network("data/shapes/COMB-Laboreiro-retinanet.shp", training)


if __name__ == '__main__':
    main()