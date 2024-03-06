import subprocess

from glob import glob



TOOLS = 'src/auxiliar/mmdetection/scripts_mamoas/tools'

def train_eval(config_file:str, work_dir:str, train_data_root:str, val_data_root:str, test_data_root:str, scores: list[float])->None:

    # Config file
    
    ann_file = 'annotations/all.json'
    
    # Train model
    # Note that train/val data_root and ann_file should be changed depending on the dataset(15/30/60) [This is to avoid having three diferent .py config files, we can modify the fields as script arguments]
    # Logs and checkpoint are saved in folds/#NUM_FOLD#    
    subprocess.run(["python3", TOOLS + "/train.py", config_file, 
                    f"--work-dir={work_dir}", 
                    "--cfg-options", 
                    f"train_dataloader.dataset.data_root={train_data_root}",
                    f"train_dataloader.dataset.ann_file={ann_file}",
                    f"val_dataloader.dataset.data_root={val_data_root}",
                    f"val_dataloader.dataset.ann_file={ann_file}",
                    f"val_evaluator.ann_file={val_data_root}{ann_file}",
                    ])


    # Get last checkpoint for testing
    lista = sorted(glob(f"{work_dir}/*.pth"))
    if len(lista)>0:
        ckpt_file = lista[-1]
    else:
        ckpt_file = None
        
    # # Test model and save predictions with different score thresholds
    # # TODO: Comment line 32 in mmdet/evaluation/metrics/dump_det_results.py: data_sample.pop('gt_instances', None)
    for score_thr in scores:
        subprocess.run(["python3", TOOLS + "/test.py", 
                                config_file, 
                                f"{ckpt_file}", 
                                # "--tta",
                                "--work-dir", f"{work_dir}",
                                "--out", f"{work_dir}/preds{score_thr}.pkl",
                                "--show-dir", f"vis_preds/",
                                f"--cfg-options",
                                f"model.test_cfg.rcnn.score_thr={score_thr}",
                                f"test_dataloader.dataset.data_root={test_data_root}",
                                f"test_dataloader.dataset.ann_file={ann_file}",
                                f"test_evaluator.ann_file={test_data_root}{ann_file}"
                                ])



        # Evaluation 
        # TODO: Comment line 32 in mmdet/evaluation/metrics/dump_det_results.py: data_sample.pop('gt_instances', None)
        # AP results are saved in resultsX.log

        eval_results = subprocess.run(["python3", TOOLS + "/eval_metric.py", 
                                        config_file, 
                                        f"{work_dir}/preds{score_thr}.pkl",
                                        f"--cfg-options",
                                        f"test_dataloader.dataset.data_root={test_data_root}",
                                        f"test_dataloader.dataset.ann_file={ann_file}",
                                        f"test_evaluator.ann_file={test_data_root}{ann_file}"]
                                        ,capture_output=True, text=True).stdout

        print("-----------------------------------")
        print("eval_results:")
        print(eval_results)
        print("-----------------------------------")

        eval_results_file = open(f"{work_dir}/results{score_thr}.log", "w")
        eval_results_file.write(eval_results)
        eval_results_file.close()

        


        # Analyze results (save images of top-k good and bad predictions in analyze folder)
        # This can be used as a reference to get the imgs with good/bad predictions, BUT
        # in order to determine if bad predictions are FN or FP, it is better to see it in
        # the images generated at vis_preds/ folder
            
        subprocess.run(["python3", TOOLS + "/analyze_results.py",
                            config_file,
                            f"{work_dir}/preds{score_thr}.pkl",
                            f"{work_dir}/analyze",
                            "--topk=20",
                            f"--show-score-thr={score_thr}",
                            "--cfg-options",
                            f"test_dataloader.dataset.data_root={test_data_root}",
                            f"test_dataloader.dataset.ann_file=annotations/all.json",
                            f"test_evaluator.ann_file={test_data_root}{ann_file}"])
            

        subprocess.run(["python3", TOOLS + "/confusion_matrix.py",
                        config_file,
                        f"{work_dir}/preds{score_thr}.pkl",
                        f"{work_dir}/confussion",
                        f"--score-thr={score_thr}",
                        "--tp-iou-thr=0.5",
                        "--color-theme=Blues"])