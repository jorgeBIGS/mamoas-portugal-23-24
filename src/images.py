
import numpy as np
import rasterio


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