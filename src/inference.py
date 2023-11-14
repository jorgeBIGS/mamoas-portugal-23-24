
import os
import shutil
from mmdet.apis import init_detector, inference_detector
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform as transform
from shapely.geometry import box
import numpy as np
import auxiliar.images as images
from PIL import Image
from parameters import *

# Specify the path to model config and checkpoint file
config_file = MODEL_PATH + '/' + MODEL + '.py'
check_point = MODEL_PATH + '/' + 'epoch_24.pth'

# Ruta al archivo TIFF georeferenciado de entrada
input_tiff = TEST_IMAGE

# Guarda el archivo shapefile
output_shapefile = ORIGINALES + '/' + SHAPE_NAME


# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint=check_point, device='cuda:0')

# Nos aseguramos de borrar todo lo que sea tiff para evitar errores no desados
shutil.rmtree(TEMPORAL, ignore_errors=True)
os.makedirs(TEMPORAL, exist_ok=True)

# Abre la imagen TIFF y genera tiles y copias sin georreferenciar.
with rasterio.open(input_tiff) as original:
    paths = images.generate_tiles(original, SIZE, OVERLAP, TEMPORAL)
    paths_no_geo = []
    for path in paths:
        path_output =  path.replace('.tif', "rgb.tif")
        paths_no_geo.append(path_output)
        images.convert_geotiff_to_tiff(path, path_output)
    
    #Clasifica y genera shapes
    
    result_shapes = []
    for image_path, image_path_no_geo in zip(paths, paths_no_geo):

        with rasterio.open(image_path) as src:
            
            data = src.read()
            profile = src.profile
            

            image = Image.open(image_path_no_geo)

            # Test a single tile and show the results
            result = inference_detector(model, np.array(image)).to_dict()

            scores = result['pred_instances']['scores'].tolist()
            bboxes = result['pred_instances']['bboxes'].tolist()


            shapes = [(transform.xy(src.transform, bbox[1], bbox[0]) 
                           + transform.xy(src.transform, bbox[3], bbox[2]),
                           score) for score, bbox in zip(scores, bboxes)]
            shapes = [(box(bbox[0], bbox[1], bbox[2], bbox[3]), score) for bbox, score in shapes]
            
            if len(shapes)>0:
                result_shapes += shapes

    if len(result_shapes)>0:   
        merged_df = pd.DataFrame(result_shapes, columns=['geometry', 'score']) 

    
        #filtramos los valores por debajo de 0.5
        merged_df = pd.DataFrame(merged_df[merged_df['score'] >= THRESHOLD])
        
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

    # Nos aseguramos de borrar todo lo que sea tiff para evitar errores de falta de disco
    shutil.rmtree(TEMPORAL, ignore_errors=True)
    os.makedirs(TEMPORAL, exist_ok=True)