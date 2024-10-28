_base_ = [
    '/data/ephemeral/home/Lv2.Object_Detection/baseline/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '/data/ephemeral/home/Lv2.Object_Detection/baseline/mmdetection/configs/_base_/default_runtime.py'
]
custom_imports = dict(imports=['mmdet.datasets.transforms'], allow_failed_imports=False)

# 이미지 크기 설정
image_size = (1024, 1024)  # 원하는 크기로 변경 가능

# Wandb logger
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'ATSS',
            'name': 'ATSS-basic-aug2',
            'entity': 'yujihwan-yonsei-university'
        }
    )
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Model settings
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=128),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,  
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),  
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # 학습 및 테스트 설정
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

# 사전 학습된 가중치 로드
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'

# Optimizer 설정
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None,
    loss_scale='dynamic'
)

# 학습률 스케줄러
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[2, 5, 8, 11],
        gamma=0.5)
]

# 기본 훅 설정
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3)
)

workflow = [('train', 1), ('val', 1)]

# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
backend_args = None

# 확률을 소수점 표현에 따른 미세한 차이 없이 설정
prob_value = 1 / 9
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True
    ),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            dict(type='ColorTransform'),
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Sharpness'),
            dict(type='Posterize'),
            dict(type='Solarize'),
            dict(type='Color'),
            dict(type='Contrast'),
            dict(type='Brightness')
        ]
    ),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True, backend='pillow'),
    # 테스트 데이터에 어노테이션이 없으므로 LoadAnnotations 제거
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',  # 'train.json'으로 수정
        data_prefix=dict(img=''),  # 이미지 경로 중복 방지를 위해 수정
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),  # 이미지 경로 중복 방지를 위해 수정
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=''),  # 이미지 경로 중복 방지를 위해 수정
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=True,  # 어노테이션이 없으므로 True로 설정
    outfile_prefix='./work_dirs/coco_detection/test',
    backend_args=backend_args
)

train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=36,  # 에포크 수를 36으로 설정
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')