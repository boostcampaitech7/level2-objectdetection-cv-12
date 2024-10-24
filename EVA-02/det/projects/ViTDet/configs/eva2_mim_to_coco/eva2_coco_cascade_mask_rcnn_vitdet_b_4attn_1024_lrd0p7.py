from functools import partial
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from ..common.coco_loader_lsj_1024 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_B_coco_bsl.pth"

model.backbone.net.img_size = 1024 
model.backbone.square_pad = 1024  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 16 
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1

# 2, 5, 8, 11 for global attention
model.backbone.net.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]

optimizer.lr=5e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None


train.max_iter = 15000
lr_multiplier.scheduler.milestones = [
    train.max_iter*8//10, train.max_iter*9//10
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.test.num_workers=0
dataloader.train.total_batch_size=128


# train
register_coco_instances(f'coco_trash_train', {}, '/data/ephemeral/home/dataset/fold/3/train.json', '/data/ephemeral/home/dataset/fold/3/')
# val
register_coco_instances(f'coco_trash_val', {}, '/data/ephemeral/home/dataset/fold/3/val.json', '/data/ephemeral/home/dataset/fold/3/')
# test
register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/dataset/test.json', '/data/ephemeral/home/dataset/')

# MetadataCatalog는 메타데이터를 설정
MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# add
dataloader.train.dataset.names = 'coco_trash_train'
dataloader.test.dataset.names = 'coco_trash_val'
dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes = False

# 체크포인트 저장 주기 설정
train.checkpointer.period = 1000

# 테스트 주기 설정
train.eval_period = 1000

# 삭제
del model.roi_heads['mask_in_features']
del model.roi_heads['mask_pooler']
del model.roi_heads['mask_head']


### batch size
dataloader.train.total_batch_size=2

# trash image data에 맞게 클래스 개수 수정
model.roi_heads.num_classes = 10