from collections import defaultdict
import rasterio 
import geopandas as gpd
import shutil
import os
from tqdm import tqdm
import os







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


def mamoas_tiles(tif_name, shapefile, size=50):

    result = defaultdict(list)

    training = get_training(shapefile)

    img = rasterio.open(tif_name)

    dst_image_dir = "data/tiles/"

    #generate_tiles(img, size, dst_image_dir)

    tile_paths = os.listdir(dst_image_dir)
    

    valid_tiles = []

    for each in tile_paths:

        img_tmp = rasterio.open(f"{dst_image_dir}/{each}")
       
      
        #print((img_tmp.width, img_tmp.height))
        rgb = img_tmp.read()

              
        
        #https://rasterio.readthedocs.io/en/latest/quickstart.html
        
        bounding_boxes = check_train(img_tmp.bounds, training)
        

        if (rgb.sum()) > 0 and len(bounding_boxes)>0:
            valid_tiles.append(each)
    
    for each in valid_tiles:
        #geotiff_to_png(f"{dst_image_dir}/{each}", f"data/valid_tiles_becario/{each}") 
        shutil.move(f"{dst_image_dir}{each}",f"data/valid_tiles/{each}")
    
    return result

if __name__ == '__main__':
    #https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
    #https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_dataset.html
    print(mamoas_tiles("data/combinacion.tif", "data/original/Mamoas-Laboreiro.shp", size=200))


    