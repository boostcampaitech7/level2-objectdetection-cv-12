# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/'

# 클래스 정보
metainfo = dict(
    classes=("General trash", "Paper", "Paper pack", "Metal", "Glass", 
             "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
)

# backend_args 설정 (필요 시 사용)
backend_args = None

# Train 및 Validation에 사용되는 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Train DataLoader 설정
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),  # data_prefix를 빈 문자열로 설정
        metainfo=metainfo,  # 클래스 정보를 metainfo로 설정
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

# Validation DataLoader 설정
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),  # data_prefix를 빈 문자열로 설정
        metainfo=metainfo,  # 클래스 정보를 metainfo로 설정
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

# Test DataLoader도 Validation과 동일하게 설정
test_dataloader = val_dataloader

# 평가자 설정 (COCO 형식의 평가 사용)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# 모델 설정 (예시)
model = dict(
    type='ATSS',  # 모델의 타입을 ATSS로 설정
    bbox_head=dict(
        num_classes=10  # 클래스 수를 10으로 설정
    )
)






