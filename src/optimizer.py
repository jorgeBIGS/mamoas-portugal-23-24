import os
import numpy
import pygad
from geopandas import GeoDataFrame 
import geopandas as gpd
from shapely.geometry import Point

TRUE_SHAPE = 'data/original/Mamoas-Laboreiro.shp'
BUFFER_SIZE = 1
SHP_DIRECTORY = 'data/shapes'

def leer_shapefiles_en_directorio(directorio):
    archivos_shp = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.shp')]

    dataframes = []

    for archivo_shp in sorted(archivos_shp):
        ruta_completa = os.path.join(directorio, archivo_shp)
        gdf = gpd.read_file(ruta_completa)
        dataframes.append(gdf)

    return dataframes


def main():
    true_positives = gpd.read_file(TRUE_SHAPE)
    
    dataframes = leer_shapefiles_en_directorio(SHP_DIRECTORY)

    desired_output = len(true_positives)
    
    
    solution, solution_fitness, solution_idx = evolve(dataframes, true_positives, desired_output)

    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

def fitness_function_factory(function_inputs, true_input, desired_output):

    def fitness_func(ga_instance, solution, solution_idx):
        fitness = 0.0
        
        for i, rectangulos in enumerate(function_inputs):
            puntajes = []
            for punto in true_input['geometry']:
                # Crea un buffer alrededor del punto para considerar una pequeña área
                buffer_punto = punto.buffer(BUFFER_SIZE)  # Ajusta el valor del buffer según tus necesidades
                
                # Encuentra los rectángulos que se superponen con el buffer del punto
                rectangulos_superpuestos = rectangulos[rectangulos.intersects(buffer_punto)]
                
                # Suma los scores de los rectángulos superpuestos que supera en threshold
                puntaje:GeoDataFrame = rectangulos_superpuestos[rectangulos_superpuestos.score >= solution[i]] 
                if not puntaje.empty:
                    puntajes.append(puntaje['score'].sum())
                else:
                    puntajes.append(0)
            
            fitness += sum(puntajes)
            total = rectangulos[rectangulos.score >= solution[i]]
            if not total.empty:
                fitness -= total['score'].sum()

        return fitness


    return fitness_func


def evolve(inputs, true_input, aim):
    fitness_function = fitness_function_factory(inputs, true_input, aim)

    num_generations = 100
    num_parents_mating = 2

    sol_per_pop = 100
    num_genes = len(inputs)

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
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
                        mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    return ga_instance.best_solution()

if __name__ == '__main__':    
    main()
