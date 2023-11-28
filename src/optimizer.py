import os
import pygad
from geopandas import GeoDataFrame 
import geopandas as gpd
import pandas as pd

from mmdetection.configs.mamoas.mamoas_detection import *

def leer_shapefiles_en_directorio(directorio):
    archivos_shp = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.shp')]

    dataframes = []

    for archivo_shp in sorted(archivos_shp):
        ruta_completa = os.path.join(directorio, archivo_shp)
        gdf = gpd.read_file(ruta_completa)
        dataframes.append(gdf)

    return dataframes

def fitness_function_factory(function_inputs:list[GeoDataFrame], true_input, desired_output):

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
    
    


    return fitness_func_3


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

def evolve(inputs, true_input, aim):
    fitness_function = fitness_function_factory(inputs, true_input, aim)

    num_generations = NUM_GENERATIONS
    num_parents_mating = NUM_PARENT_MATING

    sol_per_pop = NUM_INDIVIDUALS
    num_genes = len(inputs)

    init_range_low = 0
    init_range_high = 1

    parent_selection_type = "sss"
    keep_parents = ELITISM

    crossover_type = "single_point"

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
    ga_instance.plot_fitness()
    
    return ga_instance.best_solution()


def optimize():
    true_positives = gpd.read_file(TRUE_DATA)
    
    dataframes = leer_shapefiles_en_directorio(SHP_DIRECTORY)

    desired_output = len(true_positives) + len(dataframes)
    
    
    return evolve(dataframes, true_positives, desired_output)


if __name__ == '__main__':
    solution, solution_fitness, _ = optimize() 
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
