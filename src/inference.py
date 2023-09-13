import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from numpy import std 
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import rasterio.transform as transform
from shapely.geometry import box


# Specify the path to model config and checkpoint file
config_file = 'src/checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'src/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# Ruta al archivo TIFF georeferenciado de entrada
input_tiff = 'data/original/COMB-Arcos.tif'
# Guarda el archivo shapefile
output_shapefile = 'objetos_detectados.shp'


# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Abre la imagen TIFF
with rasterio.open(input_tiff) as src:
    # Coordenadas de la ventana de recorte (izquierda, arriba, derecha, abajo)
    
    left, top, right, bottom = (500000, 6000000, 550000, 6050000)
    # Lee la ventana de recorte
    window = Window(left, top, right - left, bottom - top)
    clipped_image = src.read(window=window)
    # Obtiene la transformación de georreferenciación de la imagen original
    clipped_transform = src.window_transform(window)

    # Test a single image and show the results
    #img = 'mmdetection/scripts_mamoas/3dogs.jpg'  # or img = mmcv.imread(img), which will only load it once
    img = mmcv.imfrombytes(clipped_image)
    result = inference_detector(model, img).to_dict()

    scores = result['pred_instances']['scores'].tolist()
    bboxes = result['pred_instances']['bboxes'].tolist()

    threshold = 3 * std(scores) + sum(scores)/len(scores)

    bboxes = [box for score, box in zip(scores, bboxes) if score>=threshold]

    data = [(x1, y1, x2, y2, 'mamoa') for x1, y1, x2, y2 in bboxes]


    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'clase'])
        
    # Convierte las coordenadas de píxeles a coordenadas geográficas utilizando 'clipped_transform'
    df['geometry'] = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(df['x1'], df['y1'], df['x2'], df['y2'])]
    df['geometry'] = df['geometry'].apply(lambda geom: transform.xy(clipped_transform, geom.bounds[1], geom.bounds[0]))

    # Crea un GeoDataFrame a partir del DataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Asigna un sistema de coordenadas de referencia (CRS) si aún no se ha hecho
    gdf.crs = src.crs

    # Guarda el archivo shapefile
    gdf.to_file(output_shapefile)
    print("Detección de objetos completada y archivo shapefile generado.")