import torch
from torchvision import transforms
from PIL import Image

# Ejemplo: cargar una imagen
image_path = 'data/mamoas-laboreiro/images/00230_.tif'
image = Image.open(image_path)

# Transformar la imagen a un tensor
transform = transforms.ToTensor()
tensor_image = transform(image)

# Obtener las dimensiones
height, width = tensor_image.shape[1], tensor_image.shape[2]

print(f"Dimensiones de la imagen: {height} x {width}")
