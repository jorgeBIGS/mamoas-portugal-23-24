
import os
from typing import List
import numpy as np
import rasterio
from tqdm import tqdm

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

def generate_tiles(tif:rasterio.io.DatasetReader, size:int, overlap: List[int], dst_dir:str)->List[str]:
    result = []
    i = 0
    for over_1 in overlap:
        for over_2 in overlap:
            for x_ini in tqdm(range(0, tif.width, size)):
                for y_ini in tqdm(range(0, tif.height, size)):
                    
                    x = x_ini + over_1
                    y = y_ini + over_2
                    
                    # creating the tile specific profile
                    profile = get_tile_profile(tif, x, y)
                    # extracting the pixel data (couldnt understand as i dont think thats the correct way to pass the argument)
                    tile_data = tif.read(window=((y, y + size), (x, x + size)),
                                        boundless=True, fill_value=profile['nodata'])[:3]
                    i+=1
                    _, dst_tile_path = get_tile_name_path(dst_dir, i)
                    c, h, w = tile_data.shape
                    profile.update(
                        height=h,
                        width=w,
                        count=c,
                        dtype=tile_data.dtype,
                    )
                    with rasterio.open(dst_tile_path, "w", **profile) as dst:
                        dst.write(tile_data)
                        result += [dst_tile_path]
    return result