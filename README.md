# Metodología para generar detector de “mamoas”

## Trabajando con los datos originales

Vamos a seguir principalmente un tutorial bastante reciente en python: https://medium.com/@kendallfrimodig/efficient-object-detection-within-satellite-imagery-using-python-85331d71ff69 y que hemos testeado en Ubuntu 22:

1. Lo primero es instalar QGIS y cargar las distintas capas raster (mdt, imágenes) y vectoriales (localización de mamoas,etc.)
2. Lo siguiente es Vectorial > Herramientas de geoproceso y seleccionar Buffer y generar una capa nueva con mini-imágenes de 1000x1000 metros u otra resolucion similar a partir de las localizaciones de mamoas para reducir a las zonas interesantes.
3. Lo siguiente es ir a Raster>Extraccion> Cortar con máscara usando el layer previamente generado. Eso hará que se genere para cada capa de entrada los “tiles” que se asocian con nuestras imágenes en una imagen cuadrada que los englobará a todos y nos ahorrará zonas que no tienen datos de mamoas.
4. Lo siguiente que toca es montar la combinación de cada raster previo. Para ello, vamos a Raster>Miscelanea>Combinar y metemos las bandas que hemos generado previamente. En principio, para entrenar las redes neuronales solo usaremos 3 bandas (RGB) por lo que si usamos imágenes con más bandas, habrá que generar falsas bandas en gris (yo he usado la ecuación Y' = 0.299 * R' + 0.587 * G' + 0.114 * B'). Es lo que he hecho con las 3 primeras bandas del linear relief model (el alpha lo he dejado fuera). 

5. A partir de ahora, cargamos el proyecto base en visual code (debe contener la imagen combinada previa)  y pasamos a python usando el repositorio (https://github.com/jorgeBIGS/mamoas-portugal-23-24.git):

- Crear environment: python -m venv .env
- source .env/bin/activate
- pip install -r config/requirements.txt

6. Nos aseguramos que existe una carpeta “data” con una combinación de 3 rasters (combinacion.tif) dentro. Ejecutamos el preprocesado y el resultado debe ser un conjunto de carpetas con las imágenes que contienes mamoas y un conjunto de ficheros json que nos ayudarán a entrenar las redes en formato COCO. Ojo que el fichero shape solo tenía POINTS de entrenamiento, cuando hubiésemos necesitado rectángulos. He generado varios shapes con cuadrados falsos de 30x30, 15x15 y 60x60  metros que potencialmente contendría las mamoas para poder hacer pruebas. Se generan en QGIS igual que en el paso 2 pero con valores 7.5,15 y 30 de “radio”.

Las carpetas anteriores nos servirán para entrenar y testear cada arquitectura de redes. El siguiente paso es pasar directamente a trabajar con la librería de detección de objetos.

## Object Detection with MMDetection

We are going to use the mmdetection library for object detection https://github.com/open-mmlab/mmdetection. 
You can find several tutorials on the documentation page (https://mmdetection.readthedocs.io/en/latest/)

Here I am just going to focus on the basics so that you can replicate what we have done. The library has many components and can be a bit overwhelming. 

We are just going to use existing object detection models, and for that we just need to put the dataset into the specified format and modify a couple of configuration files.

1. First we need to install the library https://mmdetection.readthedocs.io/en/latest/get_started.html 

    Use Python virtual env (already done) to install required libraries (requirements.txt)

    Install PyTorch following official instructions, depending on installed CUDA Version

    Install mmdetection requirements: 
    - mim install -r config/requirements_mim.txt

    You can verify the installation following this:
    
    https://mmdetection.readthedocs.io/en/latest/get_started.html#verify-the-installation

2. Generate and format data according to the library specifications (already done). Under the data folder we need to have the folder dataset (for instance, mamoas-30), and two subfolders (annotations and images)

    The images folder are all the .tif images generated by tiling the complete image. In this first stage we have used tiles of 200x200 pixels. These will be the input images of the object detection models.

    The annotations folder contains the .json files with all the information needed for training/testing the models

    It is important that annotations should follow COCO annotation format https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format. We have already taken care of this.

    Inside the annotations folder you will first see an all.json file, which contains the information of all images and bounding boxes. This file can be used for training with all images or for evaluating once we have all predictions after the cross-validation.

    Moreover, you can see a loo_cv folder, which corresponds to the .json files that will be used for the Leave-One-Out Cross Validation. For each fold number X, we have a trainX.json (with all images except one) and testX.json (with one image)

    We have already generated two datasets (with bounding boxes of 15x15 and 30x30 pixels) that can be used.

3. Once we have the dataset ready, it is time to configure an existing model so that it can be used with our dataset. All the files related to the model configuration are placed inside the src/mmdetection/configs/ folder.

    In our case, we have placed the necessary files in configs/mamoas folder. Basically, we have two config python files, one that will point to the dataset we want to use, and another one that adapts an existing model to our particular problem.

- Dataset: src/mmdetection/configs/mamoas/mamoas_detection.py. 

    This configuration file specifies which dataset is used for the experiment (data_root)

    It also specifies the train/test pipeline (which could be used for resizing the images, data augmentation, etc.) In our case, we have just fixed the resolution of input images to the original 200x200

    Finally, it specifies the train/test dataloader. This part basically points to which annotations .json is used for training and testing. For the cross-validation experiment, we will change these values dinamically when we call the script.

- Model: configs/mamoas/faster_rcnn.py

    In this file we adapt the popular Faster R-CNN model to be used for our problem. This is the config file we are going to call when we use the training/test scripts. At the beggining of the file, you can see that we import other configuration files:

    '../_base_/models/faster-rcnn_r50_fpn.py' (the default model configuration)
    '../_base_/schedules/schedule_2x.py' (the training schedule: number of epochs, learning rate, etc.)
    '../_base_/default_runtime.py' (useful configs for logging, checkpoints, etc.)
    'mamoas_detection.py' (the dataste config file that we created in the previous step)

    After these imports, what we do is to modify just some fields of the default Faster R-CNN model. In particular, the most important thing is to adjust the number of classes of our problem, which is just 1
    'roi_head=dict(
        bbox_head=dict(num_classes=1)
        ),'

    The other things (anchor_generator, score_thr, etc.) are just some hyper-parameters that could be adjusted in the future, that is why I have put them there. But there not things to worry about at the first stage.

    With these two config files, we have everything necessary to carry out experiments.

4. Train/Test models using the scripts provided in scripts_mamoas/ folder
You will find two files: train_test.sh (useful commands to get used to the library) and loo_validation.py(when everything is clear, carry out final LOO-CV experiments)

    The scripts_mamoas/train_test.sh is prepared to carry out a toy experiment with the mamoas-30 dataset using all.json for training and for testing (Note that this is not allowed in ML, and results should not be taken into account, since the testing set must be different from the training). But this is just to show the useful scripts of the library and get used to them

    You will find the following commands, more information about them inside the .sh:
    - Browse dataset to visualize input data
    - Train model
    - Test model (evaluation)
    - Different scripts to analyze results (confusion_matrix, analyze_results)

Once you know how to use the basic scripts of the library, you can proceed with the loo_validation.py script. This scripts just combines all commands explained in a loop that is repeated for each LOO experiment, then evaluates all predictions.

### Generación de shape con mamoas candidatas

1. Ejecutamos `src/preprocessing.py` con los parámetros que consideremos oportunos en `src/parameters.py`
2. Ejecutamos el script `src/trainer.py` para generar un modelo en `src/model` que usaremos para generar el shape con  los resultados.
3. El último paso es el lanzamiento de la inferencia sobre la imagen global con la que queramos trabajar. En este caso, si hemos trabajado con Laboreiro, lo normal sería probar el modelo en Arcos. Para ello, lanzamos `src/inference.py` con los parámetros adecuados en `src/parameters.py`
