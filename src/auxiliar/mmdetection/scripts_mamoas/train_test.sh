## GLOBAL VARIABLES THAT WILL BE USED FOR DIFFERENT COMMANDS
MODEL=faster_rcnn
# Config file
CONFIG_FILE=../configs/models/${MODEL}.py
# Work dir (path where checkpoints and predictions will be saved)
WORK_DIR=results/${MODEL}

# Specify data_root where images and annotations are stored
data_root="data/mamoas-laboreiro/"
# Specify path of train and val/test annotations. In this case, for simplicity, we use the same all.json for train/test.
ann_train="annotations/all.json"
ann_val="annotations/all.json"


## BROWSE DATASET: Generates .tiff images with annotation bboxes in the folder specified in the --output-dir to visualize the input data
#python3 tools/analysis_tools/browse_dataset.py \
#configs/mamoas/faster_rcnn.py \
#--output-dir vis_dataset \
#--not-show


### TRAINING
# Single-GPU training
# You need to specify the config_file(configs/mamoas/faster_rcnn.py), the work_dir where checkpoints and logs will be stored, 
# and also the ann_file for train and validation (in case it is different from the one specified in the config file configs/mamoas/mamoas_detection.py)
python3 tools/train.py ${CONFIG_FILE} --work-dir=${WORK_DIR} \
--cfg-options train_dataloader.dataset.ann_file=${ann_train} \
val_dataloader.dataset.ann_file=${ann_val} \
val_evaluator.ann_file="${data_root}${ann_val}"

# After training the model, you should go to the work_dir folder to see that the checkpoint has been generated (epoch_X.pth), and also the log file inside a subfolder with a timestamp.

#### TEST
# Once the model has been trained, we can test and evaluate to obtain the metrics. 
# The test script prints the metrics in the console and stores a log of the results. Other options are to store the results in a .pkl file if the --out arg is specified, and to plot the results if the --show-dir arg is specified.
# You need to specify the config_file(configs/mamoas/faster_rcnn.py), 
# the checkpoint of the trained model, 
# the work_dir where the log will be stored
# the --out dir where the preds.pkl will be stored
# the --show-dir where the predictions are plotted
# the cfg-options to specify a different ann-file if needed.
python3 tools/test.py ${CONFIG_FILE} ${WORK_DIR}/epoch_24.pth --work-dir=${WORK_DIR} --out=${WORK_DIR}/preds.pkl --show-dir=vis_preds \
--cfg-options test_dataloader.dataset.ann_file="${ann_val}" test_evaluator.ann_file="${data_root}${ann_val}"

# Note that any field from the original config file can be modified dynamically before calling the script with the --cfg-options args.
# After testing the model, you should see the log files inside a timestamped folder, the preds.pkl and the timestamp/vis_preds folder with images inside the result folder
# The metrics displayed are the COCO Metrics (more info at https://cocodataset.org/#detection-eval). For the moment, the most interesting metric to be considered is the Average Precision (AP) with IoU=0.5.

## USEFUL SCRIPTS FOR ANALYZING RESULTS

# The analyze_results script plots the top-k better and worse predictions. 
# You need to specify the config file, the preds.pkl where predictions are stored, and the out folder where plots will be save. The score threshold can also be specified
python3 tools/analysis_tools/analyze_results.py ${CONFIG_FILE} ${WORK_DIR}/preds.pkl  ${WORK_DIR}/analyze --show-score-thr=0.5

# The confusion matrix script can be used to generate an image with the confusion matrix
# You need to specify the config file, the preds.pkl where predictions are stored, and the out folder where plots will be save. The score and IoU threshold can also be specified
python3 tools/analysis_tools/confusion_matrix.py ${CONFIG_FILE} ${WORK_DIR}/preds.pkl  ${WORK_DIR} --score-thr=0.5 --tp-iou-thr=0.5 --color-theme=Blues


# The eval_metric script can be used if you need to evaluate the metric without having to generate the preds.pkl results again
python3 tools/analysis_tools/eval_metric.py ${CONFIG_FILE} ${WORK_DIR}/preds.pkl 