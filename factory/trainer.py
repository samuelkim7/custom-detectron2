from .evaluator import MAPIOUEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, BestCheckpointer, PeriodicWriter


class Trainer(DefaultTrainer):
    """
    Custom trainer equipped with MAPIOUEvaluator
    DATA augmentation methods implemented
    Bestcheckpointer and PeriodicWriter added
    """  
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
                   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                      T.RandomBrightness(0.9, 1.1),
                      T.RandomContrast(0.9, 1.1),
                      T.RandomSaturation(0.9, 1.1),
                      T.RandomLighting(0.9),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                   ]))
    
    def build_hooks(self):
        cfg = self.cfg.clone()
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                         DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                         "MaP IoU", 
                                         "max"))
        for hook in hooks:
            if isinstance(hook, PeriodicWriter):
                hooks.remove(hook)

        hooks.append(PeriodicWriter(self.build_writers(), period=cfg.TEST.EVAL_PERIOD))
        return hooks