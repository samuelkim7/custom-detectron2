# custom-detectron2
- Customize your training pipepline for detectron2 in a simple manner.
- You can add custom trainer, evaluator, and evaluation metrics in the factory folder
- You can add data augmentations inside your trainer implementation
- Change crucial training configurations in the configs/training.yaml

## Setup
- Install detectron2 in a conda environment (don't do pip install detectron2)
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- If there is a cuda version issue in installation, try pre-built detectron2 for linux at  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

## Run the training pipeline
- run the python script
```
bash run.sh
```
- multi-gpu training
```
bash ddp_run.sh
```
