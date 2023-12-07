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

def generate_coco_annotations(image_filenames, destiny_valid_images, train, output_file, output_directory, limites = dict()):
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
        if not image_filename in limites:
            image_width, image_height, bounds, _ = get_image_dimensions(image_filename, destiny_valid_images)
            limites[image_filename] = (image_width, image_height, bounds)
        else:
            image_width, image_height, bounds = limites[image_filename]

        image_info = {
            'id': image_id,
            'file_name': image_filename,
            'width': image_width,
            'height': image_height
        }

        images.append(image_info)

        # Genera las anotaciones para cada bounding box
        lista = check_train(bounds, train)
        bboxes_1 = []
        for bbox in lista:
            left, bottom, right, top = bbox.bounds
            w, h = max(abs(right-left), RES_MIN), max(abs(top-bottom), RES_MIN)
           
            posX = min(max(int(left-bounds.left), 0), image_width)
            posY = min(max(int(bounds.top-top), 0), image_height)

            if posX + w > image_width: w = image_width - posX 
            if posY + h > image_height: h = image_height - posY

            is_valid = False
            with rasterio.open(output_directory + image_filename) as im:
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
                    'category_id': 1,  # ID de la categoría
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

def get_image_dimensions(image_filename, root=DST_VALID_TILES_L1):
    # Aquí puedes implementar la lógica para obtener las dimensiones de la imagen
    # Por ejemplo, usando PIL o cualquier otra biblioteca de imágenes
    with rasterio.open(f"{root}{image_filename}") as dataset:
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


def check_train(tile_bounds, train):
    result = []

    for bbox in train:
        xmin, ymin, xmax, ymax = bbox.bounds[0], bbox.bounds[1], bbox.bounds[2], bbox.bounds[3]
        #TODO: Problem with the overlap between tile and bbox. By now, we focus on a minimum overlap area (LENIENT) or a complete overlap (STRICT).
        if COMPLETE_BBOX_OVERLAP:
            #STRICT 
            if check_limit(tile_bounds, xmin, ymin) and check_limit(tile_bounds, xmin, ymax) and check_limit(tile_bounds, xmax, ymin) and check_limit(tile_bounds, xmax, ymax):
                result.append(bbox)
        else:
            #LENIENT
            if check_area(tile_bounds, bbox.bounds)>=LENIENT_BBOX_OVERLAP_PERCENTAGE:
                result.append(bbox)

    return result

def get_training(shapefile:str)->list:
    result = []
    gdf = gpd.read_file(shapefile)
    
    for x in gdf.values:
        result.append(x[3])
    return result


def mamoas_tiles(tif_name:str, shapefile:str, include_all:str = INCLUDE_ALL_IMAGES, destiny_images:str= DST_IMAGE_DIR_L1, destiny_valid_images:str = DST_VALID_TILES_L1, coco_data:str = DST_DATA_IMAGES_L1, coco_data_annotation:str = DST_DATA_ANNOTATION_L1, leave_one_out:str = LEAVE_ONE_OUT_BOOL, loo_data:str = DST_DATA_LOO_CV_L1, size:int=50, overlap:List[int] = [0]):

    training = get_training(shapefile)

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
        
        bounding_boxes = check_train(img_tmp.bounds, training)
        
        #if(rgb.sum() > 0 and np.mean(rgb) !=255 and len(bounding_boxes)==0):
        #    count_background_images+=1
        #    if count_background_images>NUM_BACKGROUND_IMAGES:
        #        include=False
        #or include

        if rgb.sum() > 0 and np.mean(rgb) !=255 and (include_all or len(bounding_boxes)>0): 
            shutil.move(f"{destiny_images}{each}",f"{destiny_valid_images}{each}")
            convert_geotiff_to_tiff(f"{destiny_valid_images}{each}", f"{coco_data}{each}")
            valid_paths.append(each)

    
    info = generate_coco_annotations(valid_paths, destiny_valid_images, training, f"{coco_data_annotation}all.json", coco_data)
    
    if leave_one_out:
        for index, each in enumerate(valid_paths):
            training_set = list(valid_paths)
            training_set.remove(each)
            test_set = list()
            test_set.append(each)
            generate_coco_annotations(test_set,destiny_valid_images, training, f"{loo_data}test{index}.json", coco_data, info)
            generate_coco_annotations(training_set, destiny_valid_images, training, f"{loo_data}training{index}.json", coco_data, info)
    
    return valid_paths

def save_shape(rectangles: list, crs:CRS, name:str)->str:
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

    return name
   

if __name__ == '__main__':
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    shutil.rmtree(DST_IMAGE_DIR_L1, ignore_errors=True)
    shutil.rmtree(OUTPUT_DATA_ROOT_L1, ignore_errors=True)
    shutil.rmtree(DST_VALID_TILES_L1, ignore_errors=True)

    os.makedirs(OUTPUT_DATA_ROOT_L1, exist_ok=True)
    os.makedirs(DST_DATA_IMAGES_L1, exist_ok=True)
    os.makedirs(DST_DATA_ANNOTATION_L1, exist_ok=True)
    os.makedirs(DST_DATA_LOO_CV_L1, exist_ok=True)
    os.makedirs(DST_IMAGE_DIR_L1, exist_ok=True)
    os.makedirs(DST_VALID_TILES_L1, exist_ok=True)

    shutil.rmtree(DST_IMAGE_DIR_L2, ignore_errors=True)
    shutil.rmtree(OUTPUT_DATA_ROOT_L2, ignore_errors=True)
    shutil.rmtree(DST_VALID_TILES_L2, ignore_errors=True)

    os.makedirs(OUTPUT_DATA_ROOT_L2, exist_ok=True)
    os.makedirs(DST_DATA_IMAGES_L2, exist_ok=True)
    os.makedirs(DST_DATA_ANNOTATION_L2, exist_ok=True)
    os.makedirs(DST_DATA_LOO_CV_L2, exist_ok=True)
    os.makedirs(DST_IMAGE_DIR_L2, exist_ok=True)
    os.makedirs(DST_VALID_TILES_L2, exist_ok=True)

    shutil.rmtree(DST_IMAGE_DIR_L3, ignore_errors=True)
    shutil.rmtree(OUTPUT_DATA_ROOT_L3, ignore_errors=True)
    shutil.rmtree(DST_VALID_TILES_L3, ignore_errors=True)

    os.makedirs(OUTPUT_DATA_ROOT_L3, exist_ok=True)
    os.makedirs(DST_DATA_IMAGES_L3, exist_ok=True)
    os.makedirs(DST_DATA_ANNOTATION_L3, exist_ok=True)
    os.makedirs(DST_DATA_LOO_CV_L3, exist_ok=True)
    os.makedirs(DST_IMAGE_DIR_L3, exist_ok=True)
    os.makedirs(DST_VALID_TILES_L3, exist_ok=True)


    
    valid_images = mamoas_tiles(TRUE_IMAGE, TRUE_SHAPE, size=SIZE_L1, overlap = OVERLAP_L1)

    if len(valid_images)>0:
        save_shape([get_image_dimensions(image, DST_VALID_TILES_L1)[2] for image in valid_images], 
                   get_image_dimensions(valid_images[0], DST_VALID_TILES_L1)[3], TRUE_SHAPE.replace(".shp", "_l1.shp"))


    shutil.rmtree(DST_IMAGE_DIR_L1, ignore_errors=True)

    valid_images = mamoas_tiles(TRUE_IMAGE, TRUE_SHAPE.replace(".shp", "_l1.shp"), 
                 size = SIZE_L2,
                 overlap = OVERLAP_L2,  
                 destiny_images= DST_IMAGE_DIR_L2, 
                 destiny_valid_images = DST_VALID_TILES_L2, 
                 coco_data = DST_DATA_IMAGES_L2, 
                 coco_data_annotation = DST_DATA_ANNOTATION_L2, 
                 loo_data = DST_DATA_LOO_CV_L2
                 )
    
    if len(valid_images)>0:
        save_shape([get_image_dimensions(image, DST_VALID_TILES_L2)[2] for image in valid_images], 
                   get_image_dimensions(valid_images[0], DST_VALID_TILES_L2)[3], TRUE_SHAPE.replace(".shp", "_l2.shp"))

    shutil.rmtree(DST_IMAGE_DIR_L2, ignore_errors=True)

    valid_images = mamoas_tiles(TRUE_IMAGE, TRUE_SHAPE.replace(".shp", "_l2.shp"), 
                 size = SIZE_L3,
                 overlap = OVERLAP_L3,  
                 destiny_images= DST_IMAGE_DIR_L3, 
                 destiny_valid_images = DST_VALID_TILES_L3, 
                 coco_data = DST_DATA_IMAGES_L3, 
                 coco_data_annotation = DST_DATA_ANNOTATION_L3, 
                 loo_data = DST_DATA_LOO_CV_L3
                 )
    
    #if len(valid_images)>0:
    #    save_shape([get_image_dimensions(image, DST_VALID_TILES_L3)[2] for image in valid_images], 
    #               get_image_dimensions(valid_images[0], DST_VALID_TILES_L3)[3], TRUE_SHAPE.replace(".shp", "_l3.shp"))
    
    shutil.rmtree(DST_IMAGE_DIR_L3, ignore_errors=True)