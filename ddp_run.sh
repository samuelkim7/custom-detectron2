export PYTHONPATH=${PWD}

python main/train.py \
	--num-gpus 4 \
	--config configs/training.yaml
