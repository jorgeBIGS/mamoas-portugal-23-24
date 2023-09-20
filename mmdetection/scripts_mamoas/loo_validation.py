import os, subprocess
from glob import glob
import pickle
import json
import shutil

data_folders=["data/mamoas-30/"]

model="dynamic_rcnn"
# Config file
config_file=f"configs/mamoas/{model}.py"

# Specify the GPU you want to use (0, 1, etc.)
#gpu_id = 1
# Set the CUDA_VISIBLE_DEVICES environment variable
#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

for data_root in data_folders:
    size_bbox=data_root.split("-")[-1].replace("/","")
    work_dir=f"results_loo_{size_bbox}/{model}/" # Work dir (path where checkpoints and predictions will be saved)
    num_folds = int(len(os.listdir(f"{data_root}/annotations/loo_cv/"))/2)
    for i in range(num_folds):
        print("-----------------------------------")
        print(f"Second Loop for data_root = {data_root} with num_folds = {i}")

        work_dir_fold=f"{work_dir}folds/{i}"
        ann_train=f"annotations/loo_cv/training{i}.json"
        ann_val=f"annotations/loo_cv/test{i}.json"
        
        # Train model
        # Note that train/val data_root and ann_file should be changed depending on the dataset(15/30/60) [This is to avoid having three diferent .py config files, we can modify the fields as script arguments]
        # Logs and checkpoint are saved in folds/#NUM_FOLD#    
        subprocess.run(["python", "tools/train.py", config_file, 
                        f"--work-dir={work_dir_fold}", 
                        "--cfg-options", 
                        f"train_dataloader.dataset.data_root={data_root}",
                        f"train_dataloader.dataset.ann_file={ann_train}",
                        f"val_dataloader.dataset.data_root={data_root}",
                        f"val_dataloader.dataset.ann_file={ann_val}",
                        f"val_evaluator.ann_file={data_root}{ann_val}",
                        ])

        
        # Get last checkpoint for testing
        ckpt_file = sorted(glob(f"{work_dir_fold}/*.pth"))[-1]
        
        # Test model
        # Note that test data_root and ann_file should be changed depending on the dataset(15/30/60)
        # Predictions for each fold are saved in folds/predictions/pred#NUM_FOLD#.pkl
        # Visualization results are saved in folds/#NUM_FOLD#/timestamp_test/vis_preds/ folder (the default score_thr for visualization is 0.3)
        subprocess.run(["python", "tools/test.py", 
                        config_file, 
                        f"{ckpt_file}", 
                        "--work-dir", f"{work_dir_fold}",
                        "--out", f"{work_dir}/folds/predictions/pred{i}.pkl",
                        "--show-dir", f"vis_preds/",
                        f"--cfg-options",
                        f"test_dataloader.dataset.data_root={data_root}",
                        f"test_dataloader.dataset.ann_file={ann_val}",
                        f"test_evaluator.ann_file={data_root}{ann_val}"
                        ])

    # # Merge all predictions within folds/predictions folder into single preds.pkl for final evaluation
    labels = json.load(open(f'{data_root}annotations/all.json'))
    merged_preds = []                            
    for filename in (os.listdir(f"{work_dir}folds/predictions")):
        fold = filename.replace("pred","").replace(".pkl","")
        img_filename = json.load(open(f'{data_root}annotations/loo_cv/test{fold}.json'))['images'][0]['file_name']
        img_id = [x['id'] for x in labels['images'] if x['file_name']==img_filename][0]
        if filename.endswith(".pkl"):
            predfile = open(f"{work_dir}/folds/predictions/{filename}","rb")
            pred = pickle.load(predfile)
            pred[0]['img_id'] = img_id
            merged_preds.append(pred[0])
            predfile.close()
    
    merged_preds = sorted(merged_preds,key=lambda x:x['img_id'])
    merged_preds_file = open(f"{work_dir}/preds.pkl","wb")
    pickle.dump(merged_preds, merged_preds_file)
    merged_preds_file.close()
    
    
    # Evaluation
    # Note that test_dataloader data_root and ann_file / and val_evaluator ann_file should be changed depending on the dataset(15/30/60)
    # AP results are saved in results.log
    eval_results = subprocess.run(["python", "tools/analysis_tools/eval_metric.py", 
                                config_file, 
                                f"{work_dir}/preds.pkl",
                                f"--cfg-options",
                                f"test_dataloader.dataset.data_root={data_root}",
                                f"test_dataloader.dataset.ann_file=annotations/all.json",
                                f"val_evaluator.ann_file={data_root}annotations/all.json"]
                                ,capture_output=True, text=True).stdout

    print("-----------------------------------")
    print("eval_results:")
    print(eval_results)
    print("-----------------------------------")

    eval_results_file = open(f"{work_dir}results.log", "w")
    eval_results_file.write(eval_results)
    eval_results_file.close()
    
    
    
    # Move visualization results to combine all in single folder vis_preds/ at upper level
    vis_path = f"{work_dir}vis_preds"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    for f in glob(f"{work_dir}folds/**/*.tif",recursive=True):
        f_name = f.split("/")[-1]
        try:
            shutil.copyfile(f,f"{vis_path}/{f_name}")
        except shutil.SameFileError:
            pass
        
    # Confusion matrix (User must set score and iou threshold)
    # Note that test_dataloader data_root and ann_file should be changed depending on the dataset(15/30/60)
    # The confusion matrix and normalized confusion matrix are saved in png files
    subprocess.run(["python", "tools/analysis_tools/confusion_matrix.py",
                    config_file,
                    f"{work_dir}/preds.pkl",
                    work_dir,
                    "--score-thr=0.5",
                    "--tp-iou-thr=0.5",
                    "--color-theme=Blues",
                    "--cfg-options",
                    f"test_dataloader.dataset.data_root={data_root}",
                    f"test_dataloader.dataset.ann_file=annotations/all.json"])
        
        
    # Analyze results (save images of top-k good and bad predictions in analyze folder)
    # This can be used as a reference to get the imgs with good/bad predictions, BUT
    # in order to determine if bad predictions are FN or FP, it is better to see it in
    # the images generated at vis_preds/ folder
    
    # Note that test_dataloador data_root and ann_file should be changed depending on the dataset(15/30/60)
    
    subprocess.run(["python", "tools/analysis_tools/analyze_results.py",
                    config_file,
                    f"{work_dir}/preds.pkl",
                    f"{work_dir}/analyze",
                    "--topk=20",
                    "--show-score-thr=0.5",
                    "--cfg-options",
                    f"test_dataloader.dataset.data_root={data_root}",
                    f"test_dataloader.dataset.ann_file=annotations/all.json",
                    f"test_evaluator.ann_file={data_root}{ann_val}"])