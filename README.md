# sartorius-cell

## Setup
- Install detectron2 in a conda environment (don't do pip install detectron2)
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
- If there is a cuda version issue in installation, try pre-built detectron2 for linux at  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

## train with the jupyter notebook
- Revise the paths of dataset and coco-format json files (in the zip file)
- Run each cell in the notebook


## train with the python script
- Revise the paths accordingly
- run the python script
```
python train_net_cell.py
```
- multi-gpu training
```
python train_net_cell.py --num-gpus 4
```
