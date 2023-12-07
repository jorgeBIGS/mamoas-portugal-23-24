import os
import pygad
from geopandas import GeoDataFrame
import geopandas as gpd
import pandas as pd
import torch
from mmcv.ops import nms

from mmdetection.configs.mamoas_detection import *

SHP_DIRECTORY = 'data/shapes/retinanet'

def leer_shapefiles_en_directorio(directorio):
    archivos_shp = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.shp')]

    dataframes = []

    for archivo_shp in sorted(archivos_shp):
        ruta_completa = os.path.join(directorio, archivo_shp)
        gdf = gpd.read_file(ruta_completa)
        dataframes.append(gdf)

    return list(zip(sorted(archivos_shp), dataframes))

def fitness_function_factory(function_inputs:list[GeoDataFrame], true_input, desired_output):
    NUM_TRUE = len(true_input)

    def fitness_func_5(ga_instance, solution, solution_idx):
        fitness = 0.0
        interseccion = GeoDataFrame()
        for i, rectangulos in enumerate(function_inputs):
            if solution[i]<=solution[i+len(function_inputs)]:
                datos = rectangulos[(rectangulos['score'] >= solution[i]) & (rectangulos['score'] <= solution[i+len(function_inputs)])].copy()
                #Falta aplicar nms con el parámetro optimizado
                # Convertir la geometría a cajas delimitadoras
                bboxes = datos.geometry.bounds

                # Obtener las puntuaciones desde la columna 'score'
                scores = datos['score'].values

                # Aplicar NMS manualmente
                _, nms_indices = nms(torch.tensor(bboxes.values).float(), torch.tensor(scores).float(), iou_threshold=solution[i+2*len(function_inputs)])

                # Crear un nuevo GeoDataFrame con los resultados después de NMS
                datos = datos.iloc[nms_indices].copy()

                if len(datos)>0:
                    tam = len(datos)  
                    
                    mamoas_cubiertas = gpd.sjoin(true_input, datos, how="left", predicate='intersects')
                    mamoas_no_cubiertas = mamoas_cubiertas[mamoas_cubiertas['index_right'].isnull()]
                    
                    negativos = len(mamoas_no_cubiertas)
                    positivos = NUM_TRUE - negativos
                    
                    fitness += (2*positivos - negativos - tam)/NUM_TRUE

        return fitness



    #Probar máximo y mínimo en lugar de mínimo solamente, pero con mínimo solapamiento...
    def fitness_func_4(ga_instance, solution, solution_idx):
        fitness = float('-inf') 
        fusion:GeoDataFrame = GeoDataFrame()
        for i, rectangulos in enumerate(function_inputs):
            if solution[i]<=solution[i+len(function_inputs)]:
                datos = rectangulos[(rectangulos['score'] >= solution[i]) & (rectangulos['score'] <= solution[i+len(function_inputs)])].copy()
                if len(datos)>0:
                    fusion = pd.concat([fusion, datos], ignore_index=True)
        
        tam = len(fusion)  
    
        if tam>0:
            mamoas_cubiertas = gpd.sjoin(true_input, fusion, how="left", predicate='intersects')
            mamoas_no_cubiertas = mamoas_cubiertas[mamoas_cubiertas['index_right'].isnull()]
            
            # Cuenta las mamoas no cubiertas 
            negativos = 0
            if not mamoas_no_cubiertas.empty:
                negativos = len(mamoas_no_cubiertas)
                fitness = -negativos
            
                # Cuenta el número total de cajas generadas que no solaparon positivos o fueron redundantes
                positivos = len(true_input)- negativos
                fitness -= (tam-positivos)
            
        return fitness
    
    #Fitness que maximiza el número de cajas solapadas (misma resolución) sobre las mamoas reales y penaliza el exceso de cajas en el shape.
    def fitness_func_3(ga_instance, solution, solution_idx):
        fitness = 0.0
        fusion:GeoDataFrame = GeoDataFrame()
        for i, rectangulos in enumerate(function_inputs):
            if fusion.empty:
                fusion = rectangulos[rectangulos.score >= solution[i]].copy()
            else:
                fusion = pd.concat([fusion, rectangulos[rectangulos.score >= solution[i]]], ignore_index=True)
        
        
        mamoas_cubiertas = gpd.sjoin(true_input, fusion, how="inner", predicate='intersects')
        tam = len(fusion)  
        
        # Suma los scores de los rectángulos superpuestos que supera en threshold 
        positivos = 0
        if not mamoas_cubiertas.empty:
            positivos = len(mamoas_cubiertas)
            fitness += positivos
        
        if not fusion.empty:
            fitness -= (tam-positivos)
            
        return fitness

    #Fitness que maximiza la suma de los scores de cajas solapadas (misma resolución) sobre las mamoas reales y penaliza con el score del exceso de cajas en el shape.
    def fitness_func_2(ga_instance, solution, solution_idx):
        fitness = 0.0
        fusion:GeoDataFrame = GeoDataFrame()
        for i, rectangulos in enumerate(function_inputs):
            if fusion.empty:
                fusion = rectangulos[rectangulos.score >= solution[i]].copy()
            else:
                fusion = pd.concat([fusion, rectangulos[rectangulos.score >= solution[i]]], ignore_index=True)
        

        
        rectangulos_superpuestos = gpd.overlay(fusion, true_input, how='intersection', keep_geom_type=False)
            
        
        # Suma los scores de los rectángulos superpuestos que supera en threshold 
        positivos = 0.0
        if not rectangulos_superpuestos.empty:
            positivos = rectangulos_superpuestos['score'].sum()
            fitness += positivos
        
        if not fusion.empty:
            fitness -= (fusion['score'].sum()-positivos)
            
        return fitness
    
    


    return fitness_func_5


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

def evolve(inputs, true_input, aim):
    fitness_function = fitness_function_factory(inputs, true_input, aim)

    num_generations = NUM_GENERATIONS
    num_parents_mating = NUM_PARENT_MATING

    sol_per_pop = NUM_INDIVIDUALS
    num_genes = 3*len(inputs)

    init_range_low = 0
    init_range_high = 1

    parent_selection_type = "sss"
    keep_parents = ELITISM

    crossover_type = "uniform" #"single_point"

    mutation_type = "random"
    mutation_percent_genes = MUTATION_PERCENT

    ga_instance = pygad.GA(num_generations=num_generations,
                        on_generation= callback_gen,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        parallel_processing=NUM_THREADS)

    ga_instance.run()
    #ga_instance.plot_fitness()
    
    return ga_instance.best_solution()


def optimize():
    true_positives = gpd.read_file(TRUE_DATA)
    
    shape_y_geo = leer_shapefiles_en_directorio(SHP_DIRECTORY)

    desired_output = 1.0
    
    
    resultados = [(name, evolve([dataframe], true_positives, desired_output)) for name, dataframe in shape_y_geo]

    return resultados


if __name__ == '__main__':
    lista_soluciones = optimize()
    for name, solucion in lista_soluciones:
        print(name)
        solution, solution_fitness, _ = solucion
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
