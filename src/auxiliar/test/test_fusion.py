import geopandas as gpd


# Carga los archivos shapefile
true_positives = gpd.read_file('data/original/Mamoas-Laboreiro.shp')
shp1 = gpd.read_file('data/shapes/retinanet/COMB-LaboreiroL1-retinanet.shp')
shp2 = gpd.read_file('data/shapes/retinanet/COMB-LaboreiroL2-retinanet.shp')
shp3 = gpd.read_file('data/shapes/retinanet/COMB-LaboreiroL3-retinanet.shp')

# Realiza la intersección
interseccion = gpd.sjoin(true_positives, shp1, how="inner", predicate='intersects')
interseccion = interseccion.rename(columns={'index_right': 'geometry_1', 'score': 'score_1'})
interseccion['score_total'] = interseccion['score_1']
idx_max_score = interseccion.groupby('geometry_1')['score_total'].idxmax()
interseccion = interseccion.loc[idx_max_score]
interseccion = interseccion.drop_duplicates(subset='geometry')

interseccion = gpd.sjoin(interseccion, shp2, how="inner", predicate='intersects')
interseccion = interseccion.rename(columns={'index_right': 'geometry_2', 'score': 'score_2'})
interseccion['score_total'] = interseccion['score_1'] + interseccion['score_2'] 
idx_max_score = interseccion.groupby('geometry_2')['score_total'].idxmax()
interseccion = interseccion.loc[idx_max_score]
interseccion = interseccion.drop_duplicates(subset='geometry')

interseccion = gpd.sjoin(interseccion, shp3, how="inner", predicate='intersects')
interseccion = interseccion.rename(columns={'index_right': 'geometry_3', 'score': 'score_3'})
interseccion['score_total'] = interseccion['score_1'] + interseccion['score_2'] + interseccion['score_3']
idx_max_score = interseccion.groupby('geometry_3')['score_total'].idxmax()
interseccion = interseccion.loc[idx_max_score]
interseccion = interseccion.drop_duplicates(subset='geometry')

# Guarda el resultado en un nuevo archivo shapefile
interseccion[['geometry_1', 'score_total']].to_file('data/shapes/COMB-Laboreiro-retinanet.shp', driver='ESRI Shapefile')