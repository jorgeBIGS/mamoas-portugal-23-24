import numpy as np
import torch
from mmcv.ops import nms
import geopandas as gpd

# Supongamos que 'bboxes' es un tensor de dimensiones (N, 5), donde N es el número de detecciones.
# La primera columna representa la coordenada x del cuadro delimitador,
# la segunda columna representa la coordenada y,
# la tercera columna representa el ancho,
# la cuarta columna representa la altura,
# y la quinta columna representa la puntuación de confianza.

# 'labels' es un tensor de dimensiones (N,), que indica las etiquetas de clase para cada detección.

boxes = np.array([[49.1, 32.4, 51.0, 35.9],
                           [49.3, 32.9, 51.0, 35.3],
                           [49.2, 31.8, 51.0, 35.4],
                           [35.1, 11.5, 39.1, 15.7],
                           [35.6, 11.8, 39.3, 14.2],
                           [35.3, 11.5, 39.9, 14.5],
                           [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
    dtype=np.float32)
iou_threshold = 0.6
dets, inds = nms(boxes, scores, iou_threshold)
print('Scores', dets[:, -1])
print('Boxes', dets[:, 0:-1])
print(inds)




# Supongamos que 'gdf' es tu GeoDataFrame con una geometría y una columna 'score'.
# 'gdf' tiene una estructura similar a la siguiente:
# gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon()], score=[0.9])
gdf = gpd.read_file('data/shapes/COMB-LaboreiroL2-faster_rcnn.shp')

print(len(gdf))

# Convertir la geometría a cajas delimitadoras
bboxes = gdf.geometry.bounds

# Obtener las puntuaciones desde la columna 'score'
scores = gdf['score'].values

# Aplicar NMS manualmente
nms_result, nms_indices = nms(torch.tensor(bboxes.values).float(), torch.tensor(scores).float(), iou_threshold=0.5)

# Crear un nuevo GeoDataFrame con los resultados después de NMS
nms_gdf = gdf.iloc[nms_indices].copy()

# Imprimir el GeoDataFrame después de NMS
print(len(nms_gdf))
