#parámetros globales
ORIGINALES = 'data/original'
TRAINING_IMAGE=ORIGINALES + '/COMB-Laboreiro.tif'
TRAINING_SHAPE=ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'
TRAINING_DATA_ROOT = 'data/mamoas'
SIZE = 200
OVERLAP = [0, 100]
RES_MIN = 5
THRESHOLD = 0.5
PERCENTILE = 0.5

#parámetros de preprocessing
DST_IMAGE_DIR = TRAINING_DATA_ROOT + "/tiles/"
DST_VALID_TILES = TRAINING_DATA_ROOT + "/valid_tiles/"
DST_DATA_ANNOTATION = TRAINING_DATA_ROOT + "/annotations/"
DST_DATA_LOO_CV = DST_DATA_ANNOTATION + "loo_cv/"
DST_DATA_IMAGES = TRAINING_DATA_ROOT + "/images/"
COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5
INCLUDE_ALL_IMAGES = True

#parámetros de training
MODEL = "faster_rcnn"
MODEL_CONFIG = f"src/mmdetection/configs/mamoas/{MODEL}.py"
MODEL_PATH = 'src/model'
LEAVE_ONE_OUT_BOOL = True

#parámetros de inference
TEMPORAL = 'data/tmp'
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
SHAPE_NAME = 'objetos_detectados-laboreiro-all-in.shp'


