#!/usr/bin/env python
# coding: utf-8

# In[3]:


from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.modeling import GeneralizedRCNNWithTTA, GeneralizedRCNN
import logging
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.config import LazyConfig, instantiate
from omegaconf import OmegaConf
from detectron2.config import get_cfg
import yaml
import pandas as pd
from tqdm import tqdm
import torch
import os

# 설정 파일 로드 (LazyConfig)
cfg = LazyConfig.load("/data/ephemeral/home/bigstar/baseline/detectron2/projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x.py")
cfg.dataloader.test.dataset.names = 'coco_trash_test'

model = instantiate(cfg.model)
model.to(cfg.train.device)

# tta_cfg를 CfgNode로 생성
tta_cfg = get_cfg()
tta_cfg.TEST.AUG.ENABLED = True
tta_cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800)
tta_cfg.TEST.AUG.MAX_SIZE = 1333
tta_cfg.TEST.AUG.FLIP = True
tta_cfg.MODEL.KEYPOINT_ON = False
tta_cfg.MODEL.LOAD_PROPOSALS = False
tta_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 클래스 수에 맞게 조정
tta_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
tta_cfg.TEST.DETECTIONS_PER_IMAGE = 100

# 기존 cfg에서 필요한 설정을 복사
# tta_cfg.MODEL.WEIGHTS = cfg.model.WEIGHTS
tta_cfg.MODEL.DEVICE = cfg.train.device

model = GeneralizedRCNNWithTTA(tta_cfg, model)
model = create_ddp_model(model)

# 모델 가중치 로드
DetectionCheckpointer(model).load('/data/ephemeral/home/bigstar/baseline/detectron2/output/model_0019999.pth')

# 테스트 데이터셋 로드
test_loader = instantiate(cfg.dataloader.test)

# 모델을 평가 모드로 설정
model.eval()

# 예측 수행
prediction_strings = []
file_names = []

for data in tqdm(test_loader):
    
    prediction_string = ''
    input=data[0]
    with torch.no_grad():
        outputs = model(data)[0]['instances']  # model에 올바른 형식으로 전달
    
    # 예측 결과 처리
    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()
    
    for target, box, score in zip(targets, boxes, scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
    
    prediction_strings.append(prediction_string)
    file_names.append(input['file_name'].replace('/data/ephemeral/home/dataset/', ''))

# 제출 파일 생성
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.train.output_dir, 'submission_mvitv2_h_tta.csv'), index=False)

