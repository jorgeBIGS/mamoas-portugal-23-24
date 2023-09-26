
import os
from mmdet.apis import init_detector, inference_detector
import numpy as np
from PIL import Image
from mmdet.registry import VISUALIZERS
import mmcv

#Intersection Over Union (IoU)
IOU=0.5

# Specify the path to model config and checkpoint file
config_file = 'src/model/faster_rcnn.py'
check_point = 'src/model/epoch_24.pth'

training = 'data/data'

# Ruta temporal
temporal = training + '/detection'


# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint=check_point, device='cuda:0')

# Init visualizer
visualizer_config = model.cfg.visualizer
visualizer_config['vis_backends'][0]['save_dir']=temporal
visualizer = VISUALIZERS.build(visualizer_config)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta
print(visualizer_config)
paths_no_geo = [name for name in os.listdir(training + '/images') ]
print(paths_no_geo)
for image_path_no_geo in paths_no_geo:
    name = training + '/images/' + image_path_no_geo
    image = Image.open(name)
    image = mmcv.imconvert(np.array(image), 'bgr', 'rgb')
    # Test a single tile and show the results
    result = inference_detector(model, np.array(image))

    # Show the results
    img = mmcv.imread(name)
    img = mmcv.imconvert(img, 'bgr', 'rgb')


    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=True)
    visualizer.show()