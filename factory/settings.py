import os
import yaml
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import default_setup
from detectron2.data.datasets import register_coco_instances


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    config_yaml = args.config
    with open(config_yaml, 'r') as jp:
        config = yaml.load(jp, Loader=yaml.FullLoader)

    # register datasets
    data_dir = config['DATASET']['DATA_DIR']
    train_set = config['DATASET']['TRAIN_SET']
    val_set = config['DATASET']['VAL_SET']
    register_datasets(data_dir, train_set, val_set)

    # get the mdoel checkpoint
    cfg.MODEL_NAME = config['MODEL']['NAME']
    model_config = config['MODEL']['CONFIG']
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

    # dataset configurations
    cfg.INPUT.MASK_FORMAT = config['DATASET']['MASK_FORMAT']
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = (val_set,)
    cfg.DATALOADER.NUM_WORKERS = config['TRAINING']['NUM_WORKERS']

    # training configurations
    cfg.SOLVER.IMS_PER_BATCH = config['TRAINING']['BATCH_SIZE']
    # cfg.SOLVER.LR_SCHEDULER_NAME = config['TRAINING']['LR_SCHEDULER']
    cfg.SOLVER.BASE_LR = config['TRAINING']['LR']
    itrs_per_epoch = ( len(DatasetCatalog.get(train_set)) ) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.WARMUP_ITERS = itrs_per_epoch 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']
    cfg.TEST.EVAL_PERIOD = itrs_per_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = itrs_per_epoch
    cfg.OUTPUT_DIR = f'{config["OUTPUT_DIR"]}/{cfg.MODEL_NAME}'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_datasets(data_dir, train_set, val_set):
    """
    Register datasets as coco instances
    """
    data_path = Path(data_dir)
    register_coco_instances(train_set,{}, f'{data_dir}/json_kaggle/annotations_train_090_1.json', data_path)
    register_coco_instances(val_set,{}, f'{data_dir}/json_kaggle/annotations_val_010_1.json', data_path)