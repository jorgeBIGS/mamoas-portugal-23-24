#parámetros globales
SIZE = 200
OVERLAP = [0, 100]
RES_MIN = 5
THRESHOLD = 0.5
PERCENTILE = 0.5
ORIGINALES = 'data/original'

#parámetros de preprocessing
TRUE_IMAGE=ORIGINALES + '/COMB-Laboreiro.tif'
TRUE_SHAPE=ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'

OUTPUT_DATA_ROOT = 'data/mamoas-laboreiro_200'
DST_IMAGE_DIR = OUTPUT_DATA_ROOT + "/tiles/"
DST_VALID_TILES = OUTPUT_DATA_ROOT + "/valid_tiles/"
DST_DATA_ANNOTATION = OUTPUT_DATA_ROOT + "/annotations/"
DST_DATA_LOO_CV = DST_DATA_ANNOTATION + "loo_cv/"
DST_DATA_IMAGES = OUTPUT_DATA_ROOT + "/images/"

OUTPUT_DATA_ROOT_x10 = 'data/mamoas-laboreiro_2000'
DST_IMAGE_DIR_x10 = OUTPUT_DATA_ROOT_x10 + "/tiles/"
DST_VALID_TILES_x10 = OUTPUT_DATA_ROOT_x10 + "/valid_tiles/"
DST_DATA_ANNOTATION_x10 = OUTPUT_DATA_ROOT_x10 + "/annotations/"
DST_DATA_LOO_CV_x10 = DST_DATA_ANNOTATION_x10 + "loo_cv/"
DST_DATA_IMAGES_x10 = OUTPUT_DATA_ROOT_x10 + "/images/"

COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5
INCLUDE_ALL_IMAGES = False
#NUM_BACKGROUND_IMAGES = 100

#parámetros de training
MODEL = "faster_rcnn"
MODEL_CONFIG = f"src/mmdetection/configs/mamoas/{MODEL}.py"
MODEL_PATH = 'src/model'
LEAVE_ONE_OUT_BOOL = False

TRAINING_IMAGE=ORIGINALES + '/COMB-Laboreiro.tif'
TRAINING_SHAPE=ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'
TRAINING_DATA_ROOT = 'data/mamoas-laboreiro_200'

VAL_IMAGE=ORIGINALES + '/COMB-Arcos.tif'
VAL_SHAPE=ORIGINALES + '/Mamoas-Arcos-cuadrados-15.shp'
VAL_DATA_ROOT = 'data/mamoas-arcos_200'

#parámetros de inference
TEMPORAL = 'data/tmp'
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
SHAPE_NAME = 'objetos_detectados-laboreiro-all-in.shp'


