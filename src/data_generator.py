import json
import shutil
import os

from pyproj import CRS
from auxiliar.images import *
import geopandas as gpd
from mmdetection.configs.mamoas_detection import *
import gc
from shapely.geometry import Polygon

def check_included(bboxes, bbox):
    result = [a['bbox'] for a in bboxes]
    return len(result)==0 or  bbox['bbox'] not in result

def generate_coco_annotations(image_filenames:list[str], 
                              train, 
                              output_file:str, 
                              output_directory:str, 
                              resolution_min:int, 
                              complete_bbox:bool, 
                              percentage_cover:float, 
                              limites:dict[str, tuple[float,float,tuple[float, float, float, float]]] = dict()):
    categories = []

    # Crea la categoría "mamoa" en el archivo de anotaciones
    category = {
        'id': 0,
        'name': 'not_mamoa',
        'supercategory': 'object'
    }
    categories.append(category)
    
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
        if not image_filename in limites:
            image_width, image_height, bounds, _ = get_image_dimensions(image_filename)
            limites[image_filename] = (image_width, image_height, bounds)
        else:
            image_width, image_height, bounds = limites[image_filename]

        image_info = {
            'id': image_id,
            'file_name': image_filename.split('/')[-1],
            'width': image_width,
            'height': image_height
        }

        images.append(image_info)

        # Genera las anotaciones para cada bounding box
        lista = check_train(bounds, train, complete_bbox, percentage_cover)
        bboxes_1 = []
        for bbox,categoria in lista:
            left, bottom, right, top = bbox.bounds
            w, h = max(abs(right-left), resolution_min), max(abs(top-bottom), resolution_min)
           
            posX = min(max(int(left-bounds.left), 0), image_width)
            posY = min(max(int(bounds.top-top), 0), image_height)

            if posX + w > image_width: w = image_width - posX 
            if posY + h > image_height: h = image_height - posY

            is_valid = False
            with rasterio.open(image_filename) as im:
                # extracting the pixel data (couldnt understand as i dont think thats the correct way to pass the argument)
                tile_data = im.read(window=((posY, posY+h), (posX, posX+w)),
                                    boundless=True, fill_value=0)[:3]
                # remove filled boxes as ground truth
                is_valid = np.mean(tile_data) !=255 and np.mean(tile_data) !=0
            
            # añadido para eliminar problemas de mamoas repetidas en el shape de partida.
            aux_bbox = [posX, posY, w, h]
            aux_bbox_tuple = (image_id, aux_bbox)
            if is_valid and not aux_bbox_tuple in bboxes_1:
                annotation = {
                    'id': id_annot,
                    'image_id': image_id,
                    'category_id': categoria,  # ID de la categoría
                    'bbox': aux_bbox,
                    'area': w * h,
                    'iscrowd': 0
                }
                annotations.append(annotation)
                bboxes_1.append(aux_bbox_tuple)
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
        
    del(f)
    
    gc.collect()    
    
    return limites

def get_image_dimensions(image_filename_path):
    # Aquí puedes implementar la lógica para obtener las dimensiones de la imagen
    # Por ejemplo, usando PIL o cualquier otra biblioteca de imágenes
    with rasterio.open(image_filename_path) as dataset:
        image_width, image_height, bounds, crs = dataset.width, dataset.height, dataset.bounds, dataset.crs
    return image_width, image_height, bounds, crs

## Probado
def check_limit(bounds, x, y):
    return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top

def check_area(tile, bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    area_bbox = (xmax-xmin)*(ymax-ymin)
    xmin = tile.left if xmin <= tile.left else xmin
    xmax = tile.right if xmax >= tile.right else xmax
    ymin = tile.bottom if ymin<=tile.bottom else ymin
    ymax = tile.top if ymax>= tile.top else ymax
    if (xmax-xmin)>0 and (ymax-ymin)>0:
        result = (xmax-xmin)*(ymax-ymin)/area_bbox
    else:
        result = 0
    return result

def refine(training, true_data):
    # Cargar el shapefile que contiene los elementos para la actualización
    shp_actualizar = gpd.read_file(true_data)

     # Realizar la intersección con sufijos en los campos
    shp_interseccion = gpd.sjoin(training, shp_actualizar, how='left', predicate='intersects')

    # Actualizar el campo deseado solo donde hay solape
    shp_interseccion['es_mamoa'] = shp_interseccion['es_mamoa'].fillna(0)

    # Descartar las geometrías y campos no necesarios
    columnas_resultado = ['es_mamoa', 'geometry']
    shp_resultado = shp_interseccion[columnas_resultado]
    return shp_resultado

def check_train(tile_bounds, train, complete_bbox_overlap:bool, lenient_bbox_overlap_percentage:float):
    result = []

    # es_mamoa, geometry
    for tupla in train.values:
        es_mamoa, geometry = tupla
        xmin, ymin, xmax, ymax = geometry.bounds[0], geometry.bounds[1], geometry.bounds[2], geometry.bounds[3]
        #TODO: Problem with the overlap between tile and bbox. By now, we focus on a minimum overlap area (LENIENT) or a complete overlap (STRICT).
        if complete_bbox_overlap:
            #STRICT 
            if check_limit(tile_bounds, xmin, ymin) and check_limit(tile_bounds, xmin, ymax) and check_limit(tile_bounds, xmax, ymin) and check_limit(tile_bounds, xmax, ymax):
                result.append((geometry, es_mamoa))
        else:
            #LENIENT
            if check_area(tile_bounds, geometry.bounds)>=lenient_bbox_overlap_percentage:
                result.append((geometry, es_mamoa))

    return result

def get_training(shapefile:str)->list:
    #result = []
    gdf = gpd.read_file(shapefile)
    return gdf

    #for x in gdf.values:
    #   result.append(x)
    #return result


def mamoas_tiles(tif_name:str, 
                 shapefile:str, 
                 true_data:str,
                 include_all:bool, 
                 leave_one_out:bool, 
                 size:int, 
                 resolution_min:float,
                 overlap:list[int],
                 complete_bbox:bool,
                 percentage_cover:float, 
                 output_data_root:str)->list[str]:

    os.makedirs(output_data_root, exist_ok=True)
    
    destiny_valid_images:str = output_data_root + "valid_tiles/"
    os.makedirs(destiny_valid_images, exist_ok=True)
    
    destiny_images:str= output_data_root + "tiles/" 
    os.makedirs(destiny_images, exist_ok=True)

    coco_data:str= output_data_root + "images/" 
    os.makedirs(coco_data, exist_ok=True)
    
    coco_data_annotation:str = output_data_root + "annotations/"
    os.makedirs(coco_data_annotation, exist_ok=True)
    
    training = refine(get_training(shapefile), true_data)

    img = rasterio.open(tif_name)

    generate_tiles(img, size, overlap, destiny_images)

    tile_paths = os.listdir(destiny_images)

    valid_paths = []
    
    #include=True
    #count_background_images = 0
    
    for each in tile_paths:

        img_tmp = rasterio.open(f"{destiny_images}{each}")

        rgb = img_tmp.read()
 
        #https://rasterio.readthedocs.io/en/latest/quickstart.html
        
        bounding_boxes = check_train(img_tmp.bounds, training, complete_bbox, percentage_cover)
        
        #if(rgb.sum() > 0 and np.mean(rgb) !=255 and len(bounding_boxes)==0):
        #    count_background_images+=1
        #    if count_background_images>NUM_BACKGROUND_IMAGES:
        #        include=False
        #or include

        if rgb.sum() > 0 and np.mean(rgb) !=255 and (include_all or len(bounding_boxes)>0): 
            shutil.move(f"{destiny_images}{each}",f"{destiny_valid_images}{each}")
            convert_geotiff_to_tiff(f"{destiny_valid_images}{each}", f"{coco_data}{each}")
            valid_paths.append(f"{destiny_valid_images}{each}")

    info = generate_coco_annotations(valid_paths, training, f"{coco_data_annotation}all.json", coco_data, resolution_min, complete_bbox, percentage_cover)
    
    if leave_one_out:
        loo_data:str = output_data_root + "loo_cv/"
        os.makedirs(loo_data, exist_ok=True)
        for index, each in enumerate(valid_paths):
            training_set = list(valid_paths)
            training_set.remove(each)
            test_set = list()
            test_set.append(each)
            generate_coco_annotations(test_set, training, f"{loo_data}test{index}.json", coco_data, resolution_min, complete_bbox, percentage_cover, info)
            generate_coco_annotations(training_set, training, f"{loo_data}training{index}.json", coco_data, resolution_min, complete_bbox, percentage_cover, info)
    
    return valid_paths

def save_shape(rectangles: list, crs:CRS, name:str)->None:
    data_dicts = [{'Name': '',
               'X': (bounding_box[0] + bounding_box[2]) / 2,
               'Y': (bounding_box[1] + bounding_box[3]) / 2,
               'geometry': Polygon([(bounding_box[0], bounding_box[1]),
                                    (bounding_box[2], bounding_box[1]),
                                    (bounding_box[2], bounding_box[3]),
                                    (bounding_box[0], bounding_box[3])])} for bounding_box in rectangles]

    # Crear un GeoDataFrame con la geometría de los rectángulos
    rectangles = gpd.GeoDataFrame(data_dicts)


    # Especificar la referencia espacial (CRS)
    rectangles.crs = crs  # Puedes ajustar el código EPSG según tus necesidades

    # Guardar el GeoDataFrame como un archivo shape
    rectangles.to_file(name)
   
def generate_training_dataset(image:str, 
                             training_shape: str, 
                             true_data:str,
                             size:int, 
                             resolution_min:float,
                             overlapping: list[int], 
                             include_all_in_training:bool, 
                             generate_loo_training:bool, 
                             complete_bbox:bool,
                             percentage_cover:float,
                             output_data_root:str,
                             output_shape:str)->None:
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    shutil.rmtree(output_data_root, ignore_errors=True)
    
    valid_images = mamoas_tiles(image, training_shape, true_data, include_all_in_training, generate_loo_training, size, resolution_min, overlapping, complete_bbox, percentage_cover, output_data_root)

    if len(valid_images)>0:
        save_shape([get_image_dimensions(image)[2] for image in valid_images], 
                   get_image_dimensions(valid_images[0])[3], output_shape)
