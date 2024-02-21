
import os
from mmdet.apis import init_detector, inference_detector
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform as transform
from shapely.geometry import box
import numpy as np
import auxiliar.images.images as images
from PIL import Image
from mmcv.ops import nms


def infere(temporal:str, size:int, overlap:int,  model_name:str, model_path:str, model_config_path:str, test_image:str, cuda_device:int, output_shapefile:str, threshold_min:float=0.5, threshold_max:float=1.0, iou_threshold_min:float=0.5, check_point_file:str='last_checkpoint')->None:

    # Specify the path to model config and checkpoint file
    config_file = model_config_path + model_name + '.py'

    with open(model_path + model_name + '/' + check_point_file) as f:
        check_point = f.readline().strip()

    # Ruta al archivo TIFF georeferenciado de entrada
    input_tiff = test_image

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint=check_point, device='cuda:' + str(cuda_device))
    
    # Abre la imagen TIFF y genera tiles y copias sin georreferenciar.
    with rasterio.open(input_tiff) as original:
        data = original.read()
        min_max = list()
        min_max.append((data[0].min(), data[0].max()))
        min_max.append((data[1].min(), data[1].max()))
        min_max.append((data[2].min(), data[2].max()))
        
        paths = os.listdir(temporal)
        paths_no_geo = []
        if len(paths)==0:
            paths = images.generate_tiles(original, size, overlap, temporal)
            for path in paths:
                path_output =  path.replace('.tif', "rgb.tif")
                paths_no_geo.append(path_output)
                images.convert_geotiff_to_tiff(path, min_max, path_output)
        else:
            paths = [temporal + '/' + path for path in paths if not '_rgb' in path]
            paths_no_geo = [path.replace('.tif', "rgb.tif") for path in paths if not '_rgb' in path]
            
            
        #Clasifica y genera shapes
        result_shapes = []
        for image_path, image_path_no_geo in zip(paths, paths_no_geo):

            with rasterio.open(image_path) as src:
                
                #data = src.read()
                #profile = src.profile
            
                image = Image.open(image_path_no_geo)

                # Test a single tile and show the results
                result = inference_detector(model, np.array(image)).to_dict()
                scores = result['pred_instances']['scores']
                bboxes = result['pred_instances']['bboxes']
                labels = result['pred_instances']['labels']

                dets, indices = nms(bboxes, scores, iou_threshold_min, score_threshold= threshold_min)

                scores = dets[:, -1].tolist()
                bboxes = dets[:, 0:-1].tolist()
                labels = labels[indices].tolist()

                shapes = [(transform.xy(src.transform, bbox[1], bbox[0]) 
                            + transform.xy(src.transform, bbox[3], bbox[2]),
                            score) for score, bbox, label in zip(scores, bboxes, labels) if label == 1]
                shapes = [(box(bbox[0], bbox[1], bbox[2], bbox[3]), score) for bbox, score in shapes if threshold_max >= score >= threshold_min]
                
                if len(shapes)>0:
                    result_shapes += shapes

        if len(result_shapes)>0:   
            merged_df = pd.DataFrame(result_shapes, columns=['geometry', 'score']) 
            
            #filtramos los valores y nos quedamos con los extremadamente buenos.
            #best = merged_df['score'].quantile(PERCENTILE)
            #merged_df = pd.DataFrame(merged_df[merged_df['score'] >= best])
            
            # Crea un GeoDataFrame a partir del DataFrame
            gdf = gpd.GeoDataFrame(merged_df,  geometry='geometry', crs=src.crs)
        
            # Guarda el archivo shapefile
            gdf.to_file(output_shapefile)
            print("Detección de objetos completada y archivo shapefile generado.")
        else:
            print("No shape generated")
