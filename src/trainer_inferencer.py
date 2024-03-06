import os
import shutil
from auxiliar.inferencer import infere
from auxiliar.mmdetection.configs.mamoas_detection import *
from auxiliar.trainer import train

def train_models(models: list[str], training_data_root:str,validation_data_root:str, 
          model_config_root:str, model_path:str)->None:
    for model in models:
        try:
            train(config_file = model_config_root + model + '.py', 
                  work_dir_fold=model_path + model,training_root=training_data_root, 
                  validation_root=validation_data_root) 
        except Exception as e:
            print('Error en training de ', model)
            print(e)
            print(25*'-----------------\n')

def apply_models(models: list[str], model_config_root:str, model_path:str, 
         test_image: str, temporal:str, size:int, overlap:int, shapes_output: str, 
         score_min:float=0.5, score_max:float = 1.0, iou_threshold:float=0.05)->None:
    
    shutil.rmtree(temporal, ignore_errors=True)
    os.makedirs(temporal, exist_ok=True)
    os.makedirs(shapes_output, exist_ok=True)



    for model in models:
        # Guarda el archivo shapefile
        shape_name = test_image.split('/')[-1].replace('.tif','') + '-' + model + '.shp'
        output_shapefile = shapes_output + '/' + shape_name

        infere(temporal, size, overlap, model, model_path, model_config_root, test_image, 0, output_shapefile, threshold_min=score_min, threshold_max=score_max, iou_threshold_max=iou_threshold)



