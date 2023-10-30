import json
import rasterio 
import geopandas as gpd
import shutil
import os
import rasterio
from images import *
from parameters import *
import gc

def check_included(bboxes, bbox):
    result = [a['bbox'] for a in bboxes]
    return len(result)==0 or  bbox['bbox'] not in result

def generate_coco_annotations(image_filenames, train, output_file, limites = dict()):
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
            image_width, image_height, bounds = get_image_dimensions(image_filename)
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
        for bbox in lista:
            left, bottom, right, top = bbox.bounds
            w, h = max(abs(right-left), RES_MIN), max(abs(top-bottom), RES_MIN)
           
            posX = min(max(int(left-bounds.left), 0), image_width)
            posY = min(max(int(bounds.top-top), 0), image_height)

            if posX + w > image_width: w = image_width - posX 
            if posY + h > image_height: h = image_height - posY

            is_valid = False
            with rasterio.open(DST_DATA_IMAGES + image_filename) as im:
                # extracting the pixel data (couldnt understand as i dont think thats the correct way to pass the argument)
                tile_data = im.read(window=((posY, posY+h), (posX, posX+w)),
                                    boundless=True, fill_value=0)[:3]
                # remove filled boxes as ground truth
                is_valid = np.mean(tile_data) !=255 and np.mean(tile_data) !=0

            if is_valid:
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
        
    del(f)
    
    gc.collect()    
    
    return limites

def get_image_dimensions(image_filename):
    # Aquí puedes implementar la lógica para obtener las dimensiones de la imagen
    # Por ejemplo, usando PIL o cualquier otra biblioteca de imágenes
    with rasterio.open(f"{DST_VALID_TILES}{image_filename}") as dataset:
        image_width, image_height, bounds = dataset.width, dataset.height, dataset.bounds
    return image_width, image_height, bounds

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

def get_training(shapefile):
    result = []
    gdf = gpd.read_file(shapefile)
    
    for x in gdf.values:
        result.append(x[3])
    return result


def mamoas_tiles(tif_name, shapefile, size=50, overlap = [0]):

    training = get_training(shapefile)

    img = rasterio.open(tif_name)

    generate_tiles(img, size, overlap, DST_IMAGE_DIR)

    tile_paths = os.listdir(DST_IMAGE_DIR)

    valid_paths = []
    
    include=True
    count_background_images = 0
    
    for each in tile_paths:

        img_tmp = rasterio.open(f"{DST_IMAGE_DIR}{each}")

        rgb = img_tmp.read()

        
              
        #https://rasterio.readthedocs.io/en/latest/quickstart.html
        
        bounding_boxes = check_train(img_tmp.bounds, training)
        
        if(rgb.sum() > 0 and np.mean(rgb) !=255 and len(bounding_boxes)==0):
            count_background_images+=1
            if count_background_images>NUM_BACKGROUND_IMAGES:
                include=False
            

        if rgb.sum() > 0 and np.mean(rgb) !=255 and (INCLUDE_ALL_IMAGES or len(bounding_boxes)>0 or include):
            shutil.move(f"{DST_IMAGE_DIR}{each}",f"{DST_VALID_TILES}{each}")
            convert_geotiff_to_tiff(f"{DST_VALID_TILES}{each}", f"{DST_DATA_IMAGES}{each}")
            valid_paths.append(each)

    
    info = generate_coco_annotations(valid_paths, training, f"{DST_DATA_ANNOTATION}all.json")
    
    if LEAVE_ONE_OUT_BOOL:
        for index, each in enumerate(valid_paths):
            training_set = list(valid_paths)
            training_set.remove(each)
            test_set = list()
            test_set.append(each)
            generate_coco_annotations(test_set, training, f"{DST_DATA_LOO_CV}test{index}.json", info)
            generate_coco_annotations(training_set, training, f"{DST_DATA_LOO_CV}training{index}.json", info)   
             

if __name__ == '__main__':
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    shutil.rmtree(DST_IMAGE_DIR, ignore_errors=True)
    shutil.rmtree(TRAINING_DATA_ROOT, ignore_errors=True)
    shutil.rmtree(DST_VALID_TILES, ignore_errors=True)
    os.makedirs(DST_DATA_IMAGES, exist_ok=True)
    os.makedirs(DST_DATA_ANNOTATION, exist_ok=True)
    os.makedirs(DST_DATA_LOO_CV, exist_ok=True)
    os.makedirs(DST_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAINING_DATA_ROOT, exist_ok=True)
    os.makedirs(DST_VALID_TILES, exist_ok=True)
    
    mamoas_tiles(TRAINING_IMAGE, TRAINING_SHAPE, size=SIZE, overlap = OVERLAP)
    shutil.rmtree(DST_IMAGE_DIR, ignore_errors=True)
    