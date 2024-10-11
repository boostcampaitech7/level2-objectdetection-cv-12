#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader


# In[2]:


# Register Dataset
try: # register_coco_instances 함수를 사용해 COCO 형식의 데이터셋을 등록
    register_coco_instances('coco_trash_train', {}, '../../dataset/train.json', '../../dataset/')
except AssertionError:
    pass

try: # 
    register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', '../../dataset/')
except AssertionError:
    pass

# MetadataCatalog.get()를 통해 coco_trash_train 데이터셋의 클래스 이름을 지정
MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


# In[3]:


# config 불러오기
'''
1. get_cfg()를 호출해 기본 설정을 가져오기

2. model_zoo.get_config_file()을 사용해 미리 정의된 Faster R-CNN의 R101 FPN 3x 구성 파일을 로드(이 부분은 변경 가능)

'''



cfg = get_cfg() # detectron2에서 기본 설정을 가지고 오는 함수입니다.
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))


# In[4]:


# 학습 데이터셋을 coco_trash_train으로 설정
cfg.DATASETS.TRAIN = ('coco_trash_train',)

# 테스트 데이터셋을 coco_trash_test로 설정
cfg.DATASETS.TEST = ('coco_trash_test',)

# DataLoader에서 사용할 worker 수를 2로 설정 (병렬 데이터 로딩)
cfg.DATALOADER.NUM_WORKERS = 2  # (오타 수정: NUM_WOREKRS → NUM_WORKERS)

# # Faster R-CNN R101 FPN 3x 모델의 사전 학습된 가중치 사용
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')

# 한 번의 학습 배치에서 처리할 이미지 수를 4로 설정
cfg.SOLVER.IMS_PER_BATCH = 4

# 학습률(Learning Rate)을 0.001로 설정
cfg.SOLVER.BASE_LR = 0.001

# 학습 반복(iteration)을 최대 15,000번으로 설정
cfg.SOLVER.MAX_ITER = 15000

# 8000번째와 12000번째 반복(iteration)에서 학습률을 감소시키도록 설정
cfg.SOLVER.STEPS = (8000, 12000)

# 학습률 감소 비율을 0.005로 설정
cfg.SOLVER.GAMMA = 0.005

# 체크포인트 저장 주기를 3000번 반복마다 저장하도록 설정
cfg.SOLVER.CHECKPOINT_PERIOD = 3000

# 모델의 출력(결과) 파일을 저장할 디렉토리를 './output'으로 설정
cfg.OUTPUT_DIR = './output'

# 이미지당 ROI(Region of Interest) 샘플 수를 128로 설정 (RoI Head의 배치 크기)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# 모델의 클래스 수를 10개로 설정 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

# 평가 주기를 3000번 반복마다 평가하도록 설정 (TEST 단계)
cfg.TEST.EVAL_PERIOD = 3000


# In[5]:


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


# In[6]:


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


# In[7]:


# train
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

