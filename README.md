# custom-detectron2
- Customize your training pipepline for detectron2 in a simple manner.

## Features
- You can add custom trainer, evaluator, and evaluation metrics (in factory).
- You can add data augmentations inside your trainer implementation.
- See how to use BestCheckpointer and PeriodicWriter for detectron2 (in factory/trainer.py).
- Change the crucial training configurations easily (in configs/training.yaml).
- Sample implementations are all shown in the package.
- Create COCO dataset json formats with stratified kfold splits (in create_datasets).

## Setup
- Create a conda environment and install pytorch  
Sample running environment: python 3.7, torch 1.8.0, torchvision 0.9.0, cuda 11.2
- Install detectron2 in a conda environment
```
# don't do pip install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- If there is a cuda version issue in installation, try pre-built detectron2 for linux at  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

## Run the training pipeline
- single gpu training
```
bash run.sh
```
- multi-gpu training
```
bash ddp_run.sh
```
