import geopandas as gpd

from mmdet.models import faster

# Carga los archivos shapefile
shp1 = gpd.read_file('data/shapes/COMB-LaboreiroL1-retinanet.shp')
shp2 = gpd.read_file('data/shapes/COMB-LaboreiroL2-retinanet.shp')

# Realiza la intersección
interseccion = gpd.overlay(shp1, shp2, how='intersection')

# Guarda el resultado en un nuevo archivo shapefile
interseccion.to_file('data/shapes/COMB-Laboreiro-retinanet.shp', driver='ESRI Shapefile')