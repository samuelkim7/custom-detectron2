from utils.logger import custom_logger
from factory.trainer import Trainer
from factory.settings import setup

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, hooks, launch
from detectron2.evaluation import verify_results


def main(args):
    # basic setup
    cfg = setup(args)

    # logging training configurations
    logger = custom_logger('sartorius', f'{cfg.OUTPUT_DIR}/{cfg.MODEL_NAME}.log')
    msg = ''
    for k, v in cfg.items():
        msg += k + ': ' + str(v) + '\n'
    logger.info(f'::: Training Configurations ::: \n{msg}')

    # evaluation only
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # training
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        '--config', required=True, help='path to the config yaml file'
    )
    args = parser.parse_args()
        
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
