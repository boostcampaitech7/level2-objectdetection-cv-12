_base_ = [
    '../configs/_base_/schedules/schedule_1x.py', 
    '../configs/_base_/default_runtime.py'
]

# 모델 설정
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # 사전 학습된 가중치 링크

model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # 평균값
        std=[58.395, 57.12, 57.375],     # 표준편차
        bgr_to_rgb=True,                 # BGR을 RGB로 변환
        pad_size_divisor=128              # 패딩 크기 배수
    ),
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),  # FPN에서 사용할 출력 인덱스
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)  # 사전 학습된 가중치 로드
    ),
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5
        ),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False  # 공식 구현을 따름
        )
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,  # 클래스 수 설정
        in_channels=256,
        pred_kernel_size=1,  # DyHead 공식 구현을 따름
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5  # DyHead 공식 구현을 따름
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        )
    ),
    # 학습 및 테스트 설정
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# 옵티마이저 및 AMP 설정
optim_wrapper = dict(
    _delete_=True,  # 기존 설정 삭제
    type='AmpOptimWrapper',  # 자동 혼합 정밀도 사용
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }
    ),
    clip_grad=None,
    loss_scale='dynamic'  # 동적 손실 스케일링
)

# 학습률 스케줄러 설정
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=12,          # 총 학습 epoch 수
        eta_min=0,         # 최소 learning rate
        begin=0,
        end=12,
        by_epoch=True
    )
]

### WandB 시각화 설정 ###
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'DINO',
            'entity': 'yujihwan-yonsei-university',
            'name': 'DINO_NEWFOLD_12EPOCH'  # 예: swin-l_5scale_original_randaug_epochs6 형식으로 변경
        }
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 로깅 및 체크포인트 설정
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                # 체크포인트 저장 주기 (epoch 단위)
        save_best='auto',          # 자동으로 베스트 모델 저장
        max_keep_ckpts=3           # 최대 저장할 체크포인트 수
    )
)

workflow = [('train', 1), ('val', 1)]  # 학습과 검증을 번갈아 수행

# Test Time Augmentation (TTA) 모델 설정
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300
    )
)

img_scales = [(2000, 480), (2000, 1200)]  # TTA를 위한 이미지 스케일
tta_pipeline = [ 
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # `RandomFlip`은 `Pad` 전에 배치해야 플립 후 바운딩 박스 좌표를 올바르게 복구할 수 있음
                dict(type='RandomFlip', prob=1.0),
                dict(type='RandomFlip', prob=0.0)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=(
                        'img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction'
                    )
                )
            ]
        ]
    )
]

# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'
classes = (
    "General trash", "Paper", "Paper pack", "Metal", "Glass", 
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
)
backend_args = None

# 학습 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(2000, 480), (2000, 1200)],
        keep_ratio=True,
        backend='pillow'
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 검증 파이프라인
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(2000, 1200),
        keep_ratio=True,
        backend='pillow'
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'
        )
    )
]

# 테스트 파이프라인
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(2000, 1200),
        keep_ratio=True,
        backend='pillow'
    ),
    # Test Time Augmentation 주석 처리됨
    # dict(
    #     type='TestTimeAug',
    #     transforms=[
    #         dict(type='Resize', scale=(1333, 1280), keep_ratio=True),
    #     ]
    # ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'
        )
    )
]

# 학습 데이터로더 설정
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

# 검증 데이터로더 설정
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

# 테스트 데이터로더 설정
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
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)

# 검증 평가자 설정
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args
)

# 테스트 평가자 설정
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args
)
