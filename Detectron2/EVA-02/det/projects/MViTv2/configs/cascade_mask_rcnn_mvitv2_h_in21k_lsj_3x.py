from .cascade_mask_rcnn_mvitv2_b_3x import model, optimizer, train, lr_multiplier
from .common.coco_loader_lsj import dataloader


model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.depth = 80
model.backbone.bottom_up.num_heads = 3
model.backbone.bottom_up.last_block_indexes = (3, 11, 71, 79)
model.backbone.bottom_up.drop_path_rate = 0.6
model.backbone.bottom_up.use_act_checkpoint = True

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_H_in21k.pyth"

# trash image data에 맞게 클래스 개수 수정
model.roi_heads.num_classes = 10

# 데이터셋 등록
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

n_splits = 5

for fold_idx in range(n_splits):
    # Register Dataset
    try:
        # train_fold_{fold_idx}.json 파일을 등록하는 부분에서 f-string을 사용하여 경로를 올바르게 설정
        register_coco_instances(f'coco_trash_train_fold_{fold_idx}', {}, 
                                f'/data/ephemeral/home/dataset/train_fold_{fold_idx}.json', '/data/ephemeral/home/dataset/')
    except AssertionError:
        pass

    try:
        # val_fold_{fold_idx}.json 파일을 등록하는 부분에서도 f-string을 사용하여 경로를 올바르게 설정
        register_coco_instances(f'coco_trash_val_fold_{fold_idx}', {}, 
                                f'/data/ephemeral/home/dataset/val_fold_{fold_idx}.json', '/data/ephemeral/home/dataset/')
    except AssertionError:
        pass

try:
    register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/dataset/test.json', '/data/ephemeral/home/dataset/')
except AssertionError:
    pass

# MetadataCatalog는 메타데이터를 설정
MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# 0부터 4까지 가능, 사용하고자 하는 폴드를 설정
fold_idx = 0

dataloader.train.dataset.names = f'coco_trash_train_fold_{fold_idx}'
dataloader.test.dataset.names = f'coco_trash_val_fold_{fold_idx}'

dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False


# 훈련 이터레이션 수 설정
train.max_iter = 20000

# 체크포인트 저장 주기 설정
train.checkpointer.period = 5000

# 테스트 주기 설정
train.eval_period = 500

# 필요에 따라 학습률 스케줄러 조정
lr_multiplier.scheduler.milestones = [1, 10000, 15000]
lr_multiplier.scheduler.values=[1.0, 0.1, 0.01]

del model.roi_heads['mask_in_features']
del model.roi_heads['mask_pooler']
del model.roi_heads['mask_head']

### batch size
dataloader.train.total_batch_size=4