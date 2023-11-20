import subprocess
from mmdetection.configs.mamoas.mamoas_detection import *

model=MODEL
# Config file
config_file=MODEL_CONFIG
work_dir_fold = MODEL_PATH
data_root = TRAINING_DATA_ROOT + '/'
ann_train = 'annotations/all.json'
ann_val = 'annotations/all.json'

# Train model
# Note that train/val data_root and ann_file should be changed depending on the dataset(15/30/60) [This is to avoid having three diferent .py config files, we can modify the fields as script arguments]
# Logs and checkpoint are saved in folds/#NUM_FOLD#    
subprocess.run(["python", "src/mmdetection/scripts_mamoas/tools/train.py", config_file, 
                f"--work-dir={work_dir_fold}", 
                "--cfg-options", 
                f"train_dataloader.dataset.data_root={data_root}",
                f"train_dataloader.dataset.ann_file={ann_train}",
                f"val_dataloader.dataset.data_root={data_root}",
                f"val_dataloader.dataset.ann_file={ann_val}",
                f"val_evaluator.ann_file={data_root}{ann_val}",
                ])