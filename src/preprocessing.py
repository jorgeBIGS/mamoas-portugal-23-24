
import json
import rasterio 
import geopandas as gpd
import shutil
import os
from tqdm import tqdm
import os

import rasterio

from images import *


SIZE = 200
OVERLAP = [0, 100]

dst_image_dir = "data/tiles/"
dst_valid_tiles = "data/valid_tiles/"
dst_data_annotation = "data/data/annotations/"
dst_data_loo_cv = "data/data/annotations/loo_cv/"
dst_data_images = "data/data/images/"
RES_MIN = 5

def check_included(bboxes, bbox):
    result = [a['bbox'] for a in bboxes]
    return len(result)==0 or  bbox['bbox'] not in result

def generate_coco_annotations(image_filenames, train, output_file):
    categories = []

    # Crea la categoría "mamoa" en el archivo de anotaciones
    category = {
        'id': 1,
        'name': 'mamoa',
        'supercategory': 'object'
    }
    categories.append(category)


    images = []
    annotations = []

    id_annot = 1
    
    # Agrega información de las imágenes al objeto COCO
    for i, image_filename in enumerate(image_filenames):
        image_id = i + 1
        image_width, image_height, bounds = get_image_dimensions(image_filename)

        image_info = {
            'id': image_id,
            'file_name': image_filename,
            'width': image_width,
            'height': image_height
        }

        images.append(image_info)

        # Genera las anotaciones para cada bounding box
        lista = check_train(bounds, train)
        for bbox in lista:
            left, bottom, right, top = bbox.bounds
            w, h = max(abs(right-left), RES_MIN), max(abs(top-bottom), RES_MIN)
           
            posX = min(max(int(left-bounds.left), 0), image_width)
            posY = min(max(int(bounds.top-top), 0), image_height)

            if posX + w > image_width: w = image_width - posX 
            if posY + h > image_height: h = image_height - posY
            annotation = {
                'id': id_annot,
                'image_id': image_id,
                'category_id': 1,  # ID de la categoría
                'bbox': [posX, posY, w, h],
                'area': w * h,
                'iscrowd': 0
            }
            annotations.append(annotation)
            id_annot = id_annot + 1 
            

    # Guarda el archivo de anotaciones en formato JSON
    with open(output_file, 'w') as f:
        # Crea el objeto COCO
        coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
        }
        json.dump(coco_data, f)

def get_image_dimensions(image_filename):
    # Aquí puedes implementar la lógica para obtener las dimensiones de la imagen
    # Por ejemplo, usando PIL o cualquier otra biblioteca de imágenes
    dataset = rasterio.open(f"{dst_valid_tiles}{image_filename}")
    image_width, image_height, bounds = dataset.width, dataset.height, dataset.bounds
    return image_width, image_height, bounds

## Probado
def check_limit(bounds, x, y):
    return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top

def check_train(tile_bounds, train):
    result = []

    for bbox in train:
        xmin, ymin, xmax, ymax = bbox.bounds[0], bbox.bounds[1], bbox.bounds[2], bbox.bounds[3]
        if check_limit(tile_bounds, xmin, ymin) and check_limit(tile_bounds, xmin, ymax) and check_limit(tile_bounds, xmax, ymin) and check_limit(tile_bounds, xmax, ymax):
            result.append(bbox)

    return result

def get_training(shapefile):
    result = []
    gdf = gpd.read_file(shapefile)
    
    for x in gdf.values:
        result.append(x[3])
    return result


def mamoas_tiles(tif_name, shapefile, size=50, overlap = [0]):

    training = get_training(shapefile)

    img = rasterio.open(tif_name)

    generate_tiles(img, size, overlap, dst_image_dir)

    tile_paths = os.listdir(dst_image_dir)

    valid_paths = []
    
    for each in tile_paths:

        img_tmp = rasterio.open(f"{dst_image_dir}{each}")

        rgb = img_tmp.read()

        
              
        #https://rasterio.readthedocs.io/en/latest/quickstart.html
        
        bounding_boxes = check_train(img_tmp.bounds, training)
        

        if (rgb.sum()) > 0 and len(bounding_boxes)>0:
            shutil.move(f"{dst_image_dir}{each}",f"{dst_valid_tiles}{each}")
            convert_geotiff_to_tiff(f"{dst_valid_tiles}{each}", f"{dst_data_images}{each}")
            valid_paths.append(each)


    generate_coco_annotations(valid_paths, training, f"{dst_data_annotation}all.json")
    
    for index, each in enumerate(valid_paths):
        training_set = list(valid_paths)
        training_set.remove(each)
        test_set = list()
        test_set.append(each)
        generate_coco_annotations(test_set, training, f"{dst_data_loo_cv}test{index}.json")
        generate_coco_annotations(training_set, training, f"{dst_data_loo_cv}training{index}.json")   
             

if __name__ == '__main__':
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    shutil.rmtree('data/tiles', ignore_errors=True)
    shutil.rmtree('data/data', ignore_errors=True)
    shutil.rmtree('data/valid_tiles', ignore_errors=True)
    os.makedirs(f"data/data/images", exist_ok=True)
    os.makedirs(f"data/data/annotations", exist_ok=True)
    os.makedirs(f"data/data/annotations/loo_cv", exist_ok=True)
    os.makedirs('data/tiles', exist_ok=True)
    os.makedirs('data/data', exist_ok=True)
    os.makedirs('data/valid_tiles', exist_ok=True)
    
    mamoas_tiles("data/original/COMB-Laboreiro-Arcos.tif", "data/original/Mamoas-Laboreiro-cuadrados-15.shp", size=SIZE, overlap = OVERLAP)

    