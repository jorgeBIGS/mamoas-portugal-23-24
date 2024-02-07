import geopandas as gpd
from pandas import DataFrame

# Cargar el shapefile que contiene los elementos para la actualización
shp_actualizar = gpd.read_file('data/original/Mamoas-Laboreiro.shp')

# Cargar el shapefile original que quieres actualizar
shp_original = gpd.read_file('data/shapes/COMB-LaboreiroL1-faster_rcnn.shp')

# Realizar la intersección con sufijos en los campos

shp_interseccion = gpd.sjoin(shp_original, shp_actualizar, how='left', predicate='intersects')

# Actualizar el campo deseado solo donde hay solape
shp_interseccion['es_mamoa'] = shp_interseccion['es_mamoa'].fillna(0)

# Descartar las geometrías y campos no necesarios
columnas_resultado = ['es_mamoa', 'geometry']
shp_resultado = shp_interseccion[columnas_resultado]

# Guardar el resultado actualizado en un nuevo shapefile
shp_resultado.to_file('data/shapes/COMB-Laboreiro.shp', driver='ESRI Shapefile')
