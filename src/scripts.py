import rasterio 
import geopandas as gpd
import shutil
import os
from tqdm import tqdm


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


def get_training(shapefile):
    result = []
    gdf = gpd.read_file(shapefile)
    
    for x in gdf.values:
        result.append((x[1], x[2]))
    return result

def mamoas_tiles(tif_name, shapefile, size=50):

    training = get_training(shapefile)

    img = rasterio.open(tif_name)

    dst_image_dir = "data/tiles/"

    lista = generate_tiles(img, size, dst_image_dir)

    tile_paths = os.listdir(dst_image_dir)

    valid_tiles = []

    for each in tile_paths:

        img_tmp = rasterio.open(f"{dst_image_dir}/{each}")

        rgb = img_tmp.read()
        
        

        if (rgb.sum()) > 0 and :
            valid_tiles.append(each)
    
    for each in valid_tiles:

        shutil.move(f"{dst_image_dir}{each}",f"data/valid_tiles/{each}")

if __name__ == '__main__':
    mamoas_tiles("data/combinacion.tif", "data/original/Mamoas-Laboreiro.shp")