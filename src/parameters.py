TRAINING_IMAGE='data/original/COMB-Laboreiro.tif'
TRAINING_SHAPE='data/original/Mamoas-Laboreiro-cuadrados-15.shp'
TRAINING_DATA_ROOT = 'data/data'

MODEL = "faster_rcnn"
MODEL_CONFIG = f"src/mmdetection/configs/mamoas/{MODEL}.py"
MODEL_PATH = 'src/model'

TEMPORAL = 'data/tmp'
TEST_IMAGE = 'data/original/COMB-Laboreiro-Arcos.tif'
SHAPE_NAME = 'objetos_detectados-arcos-laboreiro.shp'

SIZE = 200
OVERLAP = [0, 100]
RES_MIN = 5
THRESHOLD = 0
PERCENTILE = 0.9

