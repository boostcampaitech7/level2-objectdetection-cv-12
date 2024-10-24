#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
import logging

# 로그 설정 ( output_log 폴더 안에서 output.log 라는 파일이 생성되고 거기서 실시간으로 진행상황이 보이게 된다.)
log_output_dir = './output_logs'
os.makedirs(log_output_dir, exist_ok=True)
log_file = os.path.join(log_output_dir, 'output.log')
logger = setup_logger(output=log_file)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.utils.events import EventWriter, get_event_storage


# ## StratifiedKFold를 통해서 train/val을 8 : 2 비율로 나누는 코드입니다.

# In[11]:


import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from detectron2.data.datasets import register_coco_instances

# 절대 경로를 사용하여 데이터셋 경로 설정
dataset_dir = '/data/ephemeral/home/Lv2.Object_Detection/dataset'
train_json_path = os.path.join(dataset_dir, 'train.json')
image_dir = os.path.join(dataset_dir,)  # 실제 이미지 경로로 수정

# COCO 형식의 train.json 로드
with open(train_json_path, 'r') as f:
    coco_data = json.load(f)

# image_id 별로 annotations를 묶기
image_to_annotations = {}
image_to_category = {}  # StratifiedKFold를 사용하기 위해 클래스 레이블 필요
for anno in coco_data['annotations']:
    image_id = anno['image_id']
    if image_id not in image_to_annotations:
        image_to_annotations[image_id] = []
    image_to_annotations[image_id].append(anno)
    # 이미지에 속한 클래스 레이블 추가 (첫 번째 어노테이션 기준으로 레이블 설정)
    if image_id not in image_to_category:
        image_to_category[image_id] = anno['category_id']

# 이미지 리스트 및 해당하는 클래스 라벨 추출
image_ids = list(image_to_annotations.keys())
image_labels = [image_to_category[image_id] for image_id in image_ids]

# StratifiedKFold 설정
n_splits = 5  # 원하는 K 값을 설정
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 원하는 fold 선택 (예: fold_idx = 0일 경우 첫 번째 fold를 validation으로 사용)
fold_idx = 0

# K-fold split 진행
for idx, (train_idx, val_idx) in enumerate(skf.split(image_ids, image_labels)):
    if idx == fold_idx:
        train_image_ids = [image_ids[i] for i in train_idx]
        val_image_ids = [image_ids[i] for i in val_idx]
        break

# Train과 Val에 해당하는 annotations 필터링
train_annotations = [anno for image_id in train_image_ids for anno in image_to_annotations[image_id]]
val_annotations = [anno for image_id in val_image_ids for anno in image_to_annotations[image_id]]

# Train과 Val에 해당하는 이미지 필터링
train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]

# train.json과 val.json 생성
train_split_path = os.path.join(dataset_dir, f'train_fold_{fold_idx}.json')
val_split_path = os.path.join(dataset_dir, f'val_fold_{fold_idx}.json')

train_data = coco_data.copy()
train_data['annotations'] = train_annotations
train_data['images'] = train_images
with open(train_split_path, 'w') as f:
    json.dump(train_data, f)

val_data = coco_data.copy()
val_data['annotations'] = val_annotations
val_data['images'] = val_images
with open(val_split_path, 'w') as f:
    json.dump(val_data, f)

# 데이터셋 등록 ( 원하는 폴드를 사용하면 된다 )
register_coco_instances(f"coco_trash_train_fold_{fold_idx}", {}, train_split_path, image_dir)
register_coco_instances(f"coco_trash_val_fold_{fold_idx}", {}, val_split_path, image_dir)


# In[12]:


cfg = get_cfg() # detectron2에서 기본 설정을 가지고 오는 함수입니다.
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))


# ## 하이퍼파라미터와 관련된 setting을 진행하는 코드입니다.

# In[13]:


# 0부터 4까지 가능, 사용하고자 하는 폴드를 설정
fold_idx = 0  

# K-fold로 나눈 데이터를 기반으로, 학습 및 검증 데이터셋 설정
cfg.DATASETS.TRAIN = (f'coco_trash_train_fold_{fold_idx}',)
cfg.DATASETS.TEST = (f'coco_trash_val_fold_{fold_idx}',)

# DataLoader에서 사용할 worker 수를 2로 설정 (병렬 데이터 로딩)
cfg.DATALOADER.NUM_WORKERS = 2 

# # Faster R-CNN R101 FPN 3x 모델의 사전 학습된 가중치 사용
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')

# 한 번의 학습 배치에서 처리할 이미지 수를 4로 설정
cfg.SOLVER.IMS_PER_BATCH = 4

# 학습률(Learning Rate)을 0.001로 설정
cfg.SOLVER.BASE_LR = 0.001

# 학습 반복(iteration)을 최대 15,000으로 줄여서 대충 15에폭 정도의 학습을 갖게 합니다.
cfg.SOLVER.MAX_ITER = 5000

# 8000번째와 12000번째 반복(iteration)에서 학습률을 감소시키도록 설정
cfg.SOLVER.STEPS = (2500, 4000)

# 학습률 감소 비율을 0.005로 설정
cfg.SOLVER.GAMMA = 0.005

# 체크포인트 저장 주기를 3000번 반복마다 저장하도록 설정
cfg.SOLVER.CHECKPOINT_PERIOD = 500

# 모델의 출력(결과) 파일을 저장할 디렉토리를 './output'으로 설정
cfg.OUTPUT_DIR = './output/test'

# 이미지당 ROI(Region of Interest) 샘플 수를 128로 설정 (RoI Head의 배치 크기)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# 모델의 클래스 수를 10개로 설정 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

# 평가 주기를 3000번 반복마다 평가하도록 설정 (TEST 단계)
cfg.TEST.EVAL_PERIOD = 500
cfg.TEST.SCORE_THRESH_TEST = 0.5  # AP@50 기준 설정


# In[6]:





# In[14]:


# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
'''
데이터 매퍼 (전처리) 설정:

MyMapper 함수는 입력 데이터에 대한 전처리 방법을 정의

이미지에 랜덤으로 수직 뒤집기, 밝기 및 대비 변환을 적용

변환된 이미지를 텐서로 변환하고 어노테이션을 조정하여 dataset_dict에 추가

'''

import detectron2.data.transforms as T

def MyMapper(dataset_dict):
    # 원본 데이터 복사하여 데이터 변형 시 원본 데이터가 손상되지 않도록 함
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # 이미지를 'BGR' 형식으로 불러옴 (Detectron2의 기본 설정은 BGR임)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    # 데이터 증강(transform) 리스트 설정
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  # 50% 확률로 이미지를 수직으로 뒤집음
        T.RandomBrightness(0.8, 1.8),  # 이미지 밝기를 랜덤으로 조정 (0.8배 ~ 1.8배)
        T.RandomContrast(0.6, 1.3)  # 이미지 대비를 랜덤으로 조정 (0.6배 ~ 1.3배)
    ]
    
    # 설정한 transform 리스트를 적용하여 이미지를 변환
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    # 변환된 이미지를 텐서(tensor) 형식으로 변환하여 dataset_dict에 저장 (Detectron2의 입력 형식에 맞춤)
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    # 어노테이션(annotations)을 변환된 이미지에 맞춰 적용 (변형된 이미지 좌표계에 맞게 재조정)
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')  # 'annotations'에서 하나씩 가져와 변환 수행
        if obj.get('iscrowd', 0) == 0  # 'iscrowd'가 0인 객체만 선택 (crowd 객체 제외)
    ]
    
    # 변환된 어노테이션을 바탕으로 'instances' 생성 (Detectron2에서 인스턴스 예측을 위한 포맷)
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    
    # 유효하지 않은 인스턴스(빈 인스턴스)를 필터링하여 제거
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    # 최종적으로 변형된 dataset_dict 반환
    return dataset_dict


# In[15]:


# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok = True)
            output_folder = './output_eval'
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# In[16]:


import wandb
from detectron2.utils.events import EventWriter, get_event_storage

class WandbWriter(EventWriter):
    def __init__(self, cfg, project=None, name=None):
        self.cfg = cfg
        self.run = wandb.init(project=project, name=name, config=cfg)
    
    def write(self):
        storage = get_event_storage()
        stats = {}
        # storage.histories()를 사용하여 메트릭 가져오기
        for k in storage.histories():
            v = storage.histories()[k].latest()
            if isinstance(v, (int, float)):
                stats[k] = v
        # 현재 학습 iteration 추가
        stats['iteration'] = storage.iter
        wandb.log(stats)
    
    def close(self):
        self.run.finish()



# In[17]:


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_AP = 0  # 최고 성능을 저장할 변수

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyMapper, sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok=True)
            output_folder = './output_eval'
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    
    # build_writers 메서드 오버라이드
    def build_writers(self):
        # 기본 writers 가져오기
        writers = super().build_writers()
        # WandbWriter 추가
        writers.append(WandbWriter(self.cfg, 
            project="detectron2", 
            name="wandb_test"))
        return writers

    def after_step(self):
        super().after_step()
        # 평가 주기에 도달하면 평가 수행
        next_iter = self.iter + 1
        if next_iter % self.cfg.TEST.EVAL_PERIOD == 0:
            results = self.test(self.cfg, self.model)
            # mAP 가져오기 (여기서는 bbox mAP를 사용)
            bbox_AP = results['bbox']['AP']
            # best_AP 갱신 및 모델 저장
            if bbox_AP > self.best_AP:
                self.best_AP = bbox_AP
                self.checkpointer.save("model_best")
                # wandb에 best mAP 기록
                wandb.log({'best_bbox_AP': self.best_AP, 'iteration': next_iter})
            
            # 예측 결과 시각화하여 wandb에 업로드
            self.visualize_predictions()

    def visualize_predictions(self):
        # 데이터셋에서 일부 이미지 선택
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        data = next(iter(val_loader))
        with torch.no_grad():
            # 모델을 평가 모드로 설정
            self.model.eval()
            predictions = self.model(data)
            # 다시 학습 모드로 복귀
            self.model.train()
        # 이미지와 예측 결과 시각화
        from detectron2.utils.visualizer import Visualizer
        import cv2
        v = Visualizer(data[0]['image'].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1],
                       MetadataCatalog.get(self.cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(predictions[0]['instances'].to('cpu'))
        result_image = v.get_image()
        # wandb에 이미지 업로드
        wandb.log({"Prediction Examples": [wandb.Image(result_image, caption="Prediction")]})


# In[18]:


# 학습 시작
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

