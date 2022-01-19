export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}

python main/train.py --config configs/training.yaml
