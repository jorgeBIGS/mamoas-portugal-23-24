from collections import defaultdict
import json
import rasterio 
import geopandas as gpd
import shutil
import os
from tqdm import tqdm
import os
import numpy as np
import rasterio
from PIL import Image

dst_image_dir = "data/tiles/"

def generate_coco_annotations(bounding_boxes, image_filenames, output_file):
    annotations = []
    categories = []

    # Crea la categoría "mamoa" en el archivo de anotaciones
    category = {
        'id': 1,
        'name': 'mamoa',
        'supercategory': 'object'
    }
    categories.append(category)

    # Genera las anotaciones para cada bounding box
    for i, bbox in enumerate(bounding_boxes):
        image_width, image_height = get_image_dimensions(image_filenames[i])
        x, y, w, h = bbox

        annotation = {
            'id': i + 1,
            'image_id': i + 1,
            'category_id': 1,  # ID de la categoría "building"
            'bbox': [x, y, w, h],
            'area': w * h,
            'iscrowd': 0
        }

        annotations.append(annotation)

    # Crea el objeto COCO
    coco_data = {
        'images': [],
        'annotations': annotations,
        'categories': categories
    }

    # Agrega información de las imágenes al objeto COCO
    for i, image_filename in enumerate(image_filenames):
        image_id = i + 1
        image_width, image_height = get_image_dimensions(image_filename)

        image_info = {
            'id': image_id,
            'file_name': image_filename,
            'width': image_width,
            'height': image_height
        }

        coco_data['images'].append(image_info)

    # Guarda el archivo de anotaciones en formato JSON
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

def get_image_dimensions(image_filename):
    # Aquí puedes implementar la lógica para obtener las dimensiones de la imagen
    # Por ejemplo, usando PIL o cualquier otra biblioteca de imágenes
    # image_width, image_height = ...
    # return image_width, image_height
    pass



## Probado

def check_train(bounds, train):
    result = []

    for bbox in train:
        x, y = bbox[0], bbox[1]
        if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
            result += bbox

    return result


def get_training(shapefile):
    result = []
    gdf = gpd.read_file(shapefile)
    
    for x in gdf.values:
        result.append(x[3].bounds)
    return result



   

def convert_geotiff_to_tiff(input_path, output_path):
    """
    Convierte un archivo GeoTIFF en un archivo TIFF estándar, escala los valores de Float32 a Byte (0-255).

    Parámetros:
        input_path (str): Ruta del archivo GeoTIFF de entrada.
        output_path (str): Ruta del archivo TIFF de salida sin información geoespacial.
    """
    with rasterio.open(input_path) as src:
        # Lee los datos y metadatos del archivo GeoTIFF
        data = src.read()
        profile = src.profile

    # Escala los valores de Float32 a Byte (0-255)
    data[0] = np.interp(data[0], (data[0].min(), data[0].max()), (0, 255))
    data[1] = np.interp(data[1], (data[1].min(), data[1].max()), (0, 255))
    data[2] = np.interp(data[2], (data[2].min(), data[2].max()), (0, 255))
    data = np.rint(data).astype(np.uint8)
    
    # Elimina cualquier referencia a la información geoespacial y las coordenadas
    profile.pop('transform', None)
    profile.pop('crs', None)
    profile.pop('affine', None)
    profile.update(dtype=rasterio.uint8)
    

    # Guarda los datos en el nuevo archivo TIFF sin información geoespacial
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)





# adapted from https://fractaldle.medium.com/satellite-images-deep-learning-spacenet-building-segmentation-a5d145a81c33
def get_tile_name_path(dst_dir:str, index:int):
    '''
    generating index specific tile name
    '''
    dst_tile_name = "{}_.tif".format(str(index).zfill(5))
    dst_tile_path = os.path.join(dst_dir, dst_tile_name)
    return dst_tile_name, dst_tile_path

def get_tile_transform(parent_transform, pixel_x:int,pixel_y:int):
    '''
    creating tile transform matrix from parent tif image
    '''
    crs_x = parent_transform.c + pixel_x * parent_transform.a
    crs_y = parent_transform.f + pixel_y * parent_transform.e
    tile_transform = rasterio.Affine(parent_transform.a, parent_transform.b, crs_x,
                                     parent_transform.d, parent_transform.e, crs_y)
    return tile_transform
    
def get_tile_profile(parent_tif:rasterio.io.DatasetReader, pixel_x:int, pixel_y:int):
    '''
    preparing tile profile
    '''
    tile_crs = parent_tif.crs
    tile_nodata = parent_tif.nodata if parent_tif.nodata is not None else 0
    tile_transform = get_tile_transform(parent_tif.transform, pixel_x, pixel_y)
    profile = dict(
                driver="GTiff",
                crs=tile_crs,
                nodata=tile_nodata,
                transform=tile_transform
            )
    return profile

def generate_tiles(tif:rasterio.io.DatasetReader, size:int, dst_dir:str):
    result = []
    i = 0
    for x in tqdm(range(0, tif.width, size)):
        for y in range(0, tif.height, size):
            # creating the tile specific profile
            profile = get_tile_profile(tif, x, y)
            # extracting the pixel data (couldnt understand as i dont think thats the correct way to pass the argument)
            tile_data = tif.read(window=((y, y + size), (x, x + size)),
                                 boundless=True, fill_value=profile['nodata'])[:3]
            i+=1
            dst_name, dst_tile_path = get_tile_name_path(dst_dir, i)
            c, h, w = tile_data.shape
            profile.update(
                height=h,
                width=w,
                count=c,
                dtype=tile_data.dtype,
            )
            with rasterio.open(dst_tile_path, "w", **profile) as dst:
                dst.write(tile_data)
                result += dst_tile_path
    return result

def mamoas_tiles(tif_name, shapefile, size=50):

    training = get_training(shapefile)

    img = rasterio.open(tif_name)

    generate_tiles(img, size, dst_image_dir)

    tile_paths = os.listdir(dst_image_dir)

    for each in tile_paths:

        img_tmp = rasterio.open(f"{dst_image_dir}/{each}")
       
        #print((img_tmp.width, img_tmp.height))
        rgb = img_tmp.read()
              
        #https://rasterio.readthedocs.io/en/latest/quickstart.html
        
        bounding_boxes = check_train(img_tmp.bounds, training)
        

        if (rgb.sum()) > 0 and len(bounding_boxes)>0:
            shutil.move(f"{dst_image_dir}{each}",f"data/valid_tiles/{each}")
            convert_geotiff_to_tiff(f"data/valid_tiles/{each}", f"data/dl/{each}")

if __name__ == '__main__':
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    os.makedirs('data/tiles', exist_ok=True)
    os.makedirs('data/dl', exist_ok=True)
    os.makedirs('data/valid_tiles', exist_ok=True)
    print(mamoas_tiles("data/combinacion.tif", "data/original/Mamoas-Laboreiro.shp", size=200))


    