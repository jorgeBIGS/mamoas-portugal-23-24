
import os
from mmdet.apis import init_detector, inference_detector
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform as transform
from shapely.geometry import box
import numpy as np
import auxiliar.images as images
from PIL import Image

from mmdetection.configs.mamoas.mamoas_detection import LEVEL, OVERLAP, SHAPES_OUTPUT, SIZE, TEMPORAL

def infere(model_name, model_path, model_config_path, test_image, threshold, check_point_file:str='last_checkpoint')->None:

    # Specify the path to model config and checkpoint file
    config_file = model_config_path + model_name + '.py'

    with open(model_path + model_name + '/' + check_point_file) as f:
        check_point = f.readline().strip()

    # Ruta al archivo TIFF georeferenciado de entrada
    input_tiff = test_image

    

    # Guarda el archivo shapefile
    SHAPE_NAME = test_image.split('/')[-1].replace('.tif','') + str(LEVEL) + '-' + model_name + '.shp'
    output_shapefile = SHAPES_OUTPUT + '/' + SHAPE_NAME


    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint=check_point, device='cuda:0')

    # Abre la imagen TIFF y genera tiles y copias sin georreferenciar.
    with rasterio.open(input_tiff) as original:
        paths = os.listdir(TEMPORAL)
        paths_no_geo = []
        if len(paths)==0:
            paths = images.generate_tiles(original, SIZE, OVERLAP, TEMPORAL)
            for path in paths:
                path_output =  path.replace('.tif', "rgb.tif")
                paths_no_geo.append(path_output)
                images.convert_geotiff_to_tiff(path, path_output)
        else:
            paths = [TEMPORAL + '/' + path for path in paths if not '_rgb' in path]
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

                scores = result['pred_instances']['scores'].tolist()
                bboxes = result['pred_instances']['bboxes'].tolist()


                shapes = [(transform.xy(src.transform, bbox[1], bbox[0]) 
                            + transform.xy(src.transform, bbox[3], bbox[2]),
                            score) for score, bbox in zip(scores, bboxes)]
                shapes = [(box(bbox[0], bbox[1], bbox[2], bbox[3]), score) for bbox, score in shapes if score >= threshold]
                
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
