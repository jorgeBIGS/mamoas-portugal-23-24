
import os
import shutil
from mmdet.apis import init_detector, inference_detector
from numpy import std 
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform as transform
from shapely.geometry import box
import numpy as np
import images
from PIL import Image
from preprocessing import OVERLAP, SIZE

# Specify the path to model config and checkpoint file
config_file = 'mmdetection/configs/mamoas/faster_rcnn.py'
check_point = None #'src/detection/epoch_24.pth'

# Ruta al archivo TIFF georeferenciado de entrada
input_tiff = 'data/original/COMB-Arcos.tif'

# Ruta temporal
temporal = 'data/detection'

# Guarda el archivo shapefile
output_shapefile = temporal + '/objetos_detectados.shp'


# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint=check_point, device='cuda:0')

# Nos aseguramos de borrar todo lo que sea tiff para evitar errores no desados
shutil.rmtree(temporal, ignore_errors=True)
os.makedirs(temporal, exist_ok=True)

# Abre la imagen TIFF y genera tiles y copias sin georreferenciar.
with rasterio.open(input_tiff) as original:
    paths = images.generate_tiles(original, SIZE, OVERLAP, temporal)
    paths_no_geo = []
    for path in paths:
        path_output = path.replace(".tif","rgb.tif")
        paths_no_geo.append(path_output)
        images.convert_geotiff_to_tiff(path, path_output)
    
    #Clasifica y genera shapes
    merged_df = pd.DataFrame([], columns=['x1', 'y1', 'x2', 'y2', 'score', 'geometry']) 
    for image_path, image_path_no_geo in zip(paths, paths_no_geo):

        with rasterio.open(image_path) as src:
            
            data = src.read()
            profile = src.profile
            

            image = Image.open(image_path_no_geo)

            # Test a single tile and show the results
            result = inference_detector(model, np.array(image)).to_dict()

            scores = result['pred_instances']['scores'].tolist()
            bboxes = result['pred_instances']['bboxes'].tolist()


            shapes = [(bbox[0], bbox[1], bbox[2], bbox[3], score) for score, bbox in zip(scores, bboxes)]
        
            if len(shapes)>0:   
                df = pd.DataFrame(shapes, columns=['x1', 'y1', 'x2', 'y2', 'score'])

                # Convierte las coordenadas de píxeles a coordenadas geográficas utilizando 'clipped_transform'
                df['geometry'] = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(df['x1'], df['y1'], df['x2'], df['y2'])]
                df['geometry'] = df['geometry'].apply(lambda geom: transform.xy(src.transform, geom.bounds[1], geom.bounds[0]) + 
                                                    transform.xy(src.transform, geom.bounds[3], geom.bounds[2]))
                df['geometry'] = df['geometry'].apply(lambda tupla: box(tupla[0], tupla[1], tupla[2], tupla[3]))

                #filtramos los valores
                #threshold = 3 * df['score'].std() + df['score'].mean()
                #df = pd.DataFrame(df[df['score'] >= threshold])
                
                merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    #filtramos los valores
    threshold = 3 * merged_df['score'].std() + merged_df['score'].mean()
    merged_df = pd.DataFrame(merged_df[merged_df['score'] >= threshold])
    
    # Crea un GeoDataFrame a partir del DataFrame
    gdf = gpd.GeoDataFrame(merged_df,  geometry='geometry', crs=src.crs)
    
    # Guarda el archivo shapefile
    gdf.to_file(output_shapefile)
    print("Detección de objetos completada y archivo shapefile generado.")