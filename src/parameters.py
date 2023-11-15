#parámetros globales
SIZE = 200
OVERLAP = [0, 100]
MODEL_PATH_x1 = 'data/model_x1'

SIZE_x10 = 500
OVERLAP_x10 = [0, 250]
MODEL_PATH_x10 = 'data/model_x10'

RES_MIN = 5
THRESHOLD = 0.5
PERCENTILE = 0.5
ORIGINALES = 'data/original'

LEAVE_ONE_OUT_BOOL = False
INCLUDE_ALL_IMAGES = False

#parámetros de preprocessing
TRUE_IMAGE=ORIGINALES + '/COMB-Laboreiro.tif'
TRUE_SHAPE=ORIGINALES + '/Mamoas-Laboreiro-cuadrados-15.shp'

OUTPUT_DATA_ROOT = 'data/mamoas-laboreiro_x1'
DST_IMAGE_DIR = OUTPUT_DATA_ROOT + "/tiles/"
DST_VALID_TILES = OUTPUT_DATA_ROOT + "/valid_tiles/"
DST_DATA_ANNOTATION = OUTPUT_DATA_ROOT + "/annotations/"
DST_DATA_LOO_CV = DST_DATA_ANNOTATION + "loo_cv/"
DST_DATA_IMAGES = OUTPUT_DATA_ROOT + "/images/"

OUTPUT_DATA_ROOT_x10 = 'data/mamoas-laboreiro_x10'
DST_IMAGE_DIR_x10 = OUTPUT_DATA_ROOT_x10 + "/tiles/"
DST_VALID_TILES_x10 = OUTPUT_DATA_ROOT_x10 + "/valid_tiles/"
DST_DATA_ANNOTATION_x10 = OUTPUT_DATA_ROOT_x10 + "/annotations/"
DST_DATA_LOO_CV_x10 = DST_DATA_ANNOTATION_x10 + "loo_cv/"
DST_DATA_IMAGES_x10 = OUTPUT_DATA_ROOT_x10 + "/images/"

COMPLETE_BBOX_OVERLAP=False
LENIENT_BBOX_OVERLAP_PERCENTAGE = 0.5

#parámetros de training
MODEL = "faster_rcnn"

MODEL_CONFIG = f"src/mmdetection/configs/mamoas/{MODEL}.py"

MODEL_PATH = MODEL_PATH_x10

TRAINING_DATA_ROOT = OUTPUT_DATA_ROOT_x10

VAL_DATA_ROOT = OUTPUT_DATA_ROOT_x10

#parámetros de inference
CHECK_POINT_FILE = 'epoch_12.pth'
TEMPORAL = 'data/tmp'
TEST_IMAGE = 'data/original/COMB-Laboreiro.tif'
SHAPE_NAME = 'objetos_detectados-laboreiro-all-in_x10.shp'


