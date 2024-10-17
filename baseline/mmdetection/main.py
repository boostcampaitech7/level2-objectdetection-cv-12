# Commons
import os
import argparse
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

# Test
from mmdet.engine.hooks.utils import trigger_visualization_hook

def main(args):
    # Reduce the number of repeated compilations and improve
    # training/testing speed.
    setup_cache_size_limit_of_dynamo()
    
    # config file 불러오기
    cfg = Config.fromfile(args.config_dir)

    if args.output is not None:
        cfg.work_dir = args.output
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(args.config_dir))[0])
    
    if args.train:  # train mode
        # wandb를 사용하기 위한 hook 설정
        if args.wandb_name is not None:
            cfg.visualizer.vis_backends = [
                dict(
                    type='WandbVisBackend',
                    init_kwargs=dict(
                        project='DINO',
                        name=args.wandb_name),
                )
            ]
        else:
            # wandb_name이 없을 경우 기본 LocalVisBackend 사용
            cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]
        
        # enable automatic-mixed-precision training
        if args.amp:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
            
        if args.root is not None:
            cfg.data_root = args.root
            # train, val 데이터의 data_root도 업데이트
            cfg.train_dataloader.dataset.data_root = args.root
            cfg.val_dataloader.dataset.data_root = args.root
        if args.annotation is not None:
            cfg.train_dataloader.dataset.ann_file = args.annotation  # train json 정보
        if args.valid_annotation is not None:
            cfg.val_dataloader.dataset.ann_file = args.valid_annotation  # validation json 정보
        if args.load_from is not None:
            cfg.load_from = args.load_from
        
        # 학습 시 SubmissionHook 제거
        if 'custom_hooks' in cfg:
            cfg.custom_hooks = [hook for hook in cfg.custom_hooks if hook['type'] != 'SubmissionHook']
        
    else:  # test(inference) mode
        if args.load_from is not None:
            cfg.load_from = args.load_from
        else:
            cfg.load_from = os.path.join(cfg.work_dir, f"{args.checkpoint}.pth")
        
        if args.root is not None:
            cfg.data_root = args.root
            cfg.test_dataloader.dataset.data_root = args.root
        if args.annotation is not None:
            cfg.test_dataloader.dataset.ann_file = args.annotation  # test.json 정보
        if args.output is not None:
            cfg.test_evaluator.outfile_prefix = args.output
        
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    if args.train:
        runner.train()  # start training
    else:
        runner.test()  # start testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # amp 사용여부 (store_true로 변경)
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    # 사전학습 가중치 가져오기 (기본값 None)
    parser.add_argument(
        '--load_from',
        type=str,
        default=None,
        help='load pre-trained model weight path, endswith:.pth')
    # 데이터셋 위치 (기본값 None)
    parser.add_argument(
        "--root", 
        type=str, 
        default=None,
        help="dataset's location")
    # Annotation 파일 (학습 파일) 정보 (기본값 None)
    parser.add_argument(
        "--annotation", 
        type=str, 
        default=None,
        help="annotation file name")
    parser.add_argument(
        "--valid_annotation", 
        type=str, 
        default=None,
        help="validation annotation file name")
    # output 위치 (기본값 None)
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="output's location")
    # train/test mode
    parser.add_argument(
        "--train", 
        type=int, 
        default=1,
        help="set inference/train mode, 0: inference / 1: train")
    # valid/submission mode
    parser.add_argument(
        "--valid", 
        type=int, 
        default=1,
        help="set submission/valid mode, 0: submission / 1: valid")
    # Config file (필수 인자)
    parser.add_argument(
        "--config_dir", 
        type=str, 
        required=True,
        help="config file's location")
    # wandb name
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default=None,
        help="name of wandb test name")
    #################### TEST ####################
    # model pth name
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="latest",
        help="name of checkpoint want to inference")
    # Confidence Threshold
    parser.add_argument(
        "--conf_threshold", 
        type=float, 
        default=0.3,
        help="Confidence threshold used in confusion matrix")
    # IOU Threshold
    parser.add_argument(
        "--iou_threshold", 
        type=float, 
        default=0.5,
        help="IoU threshold used in confusion matrix and mAP")

    args = parser.parse_args()

    if args.train and args.wandb_name is None:
        raise Exception("Import wandb test name")
    if args.output == "./work_dirs/default":
        print(
            "Warning: Your output directory is set to (./work_dirs/default), you should change your output directory."
        )
    if args.checkpoint == "latest":
        print(
            "Warning: Your model name is set to (latest). If not intended, change your model name."
        )

    print(args)  # 내가 설정한 arguments
    
    main(args)
