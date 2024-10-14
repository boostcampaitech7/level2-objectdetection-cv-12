auto_scale_lr = dict(base_batch_size=16)
backend_args = None
custom_hooks = [
    dict(type='SubmissionHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.engine.hooks.submission_hook',
    ])
data_root = '/data/ephemeral/home/Lv2.Object_Detection/test_dir/2'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_begin=3,
        save_best='coco/bbox_mAP_50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 20
metainfo = dict(
    classes=(
        'General trash',
        'Paper',
        'Paper pack',
        'Metal',
        'Glass',
        'Plastic',
        'Styrofoam',
        'Plastic bag',
        'Battery',
        'Clothing',
    ),
    palette=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 230),
        (106, 0, 228),
        (60, 20, 220),
        (0, 80, 100),
        (0, 0, 70),
        (50, 0, 192),
        (250, 170, 30),
        (255, 0, 0),
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=4.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=10,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[110.07, 117.39, 123.65],
        pad_size_divisor=1,
        std=[54.77, 53.35, 54.01],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5)),
        num_layers=6),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],  # 백본의 각 계층에서 나오는 feature map의 채널 크기
        out_channels=256,                   # 모든 출력 feature map의 채널 크기
        num_outs=5,                         # 생성할 피라미드 단계의 수
        start_level=0,                      # 피라미드 생성을 시작할 백본 계층
        add_extra_convs='on_input',         # 피라미드에 추가적인 컨볼루션 적용 여부
        relu_before_extra_convs=True        # 추가 컨볼루션 전에 ReLU 적용 여부
    ),
    num_feature_levels=5,
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
num_levels = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=8.3e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
# 기존 MultiStepLR 스케줄러 삭제 및 CosineAnnealingLR로 변경
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,       # 최소 학습률 설정
        begin=0,           # 학습 시작 시점
        T_max=max_epochs,  # 전체 에포크 수에 따라 학습률 감소
        end=max_epochs,    # 학습이 끝날 때까지 적용
        by_epoch=True,     # 에포크 단위로 학습률 스케줄 적용
        convert_to_iter_based=True  # 에포크 기반을 반복(iteration) 기반으로 변환
    )
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
randomness = dict(seed=442310925)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/data/ephemeral/home/Lv2.Object_Detection/dataset/test.json',
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(1024, 1024), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/data/ephemeral/home/Lv2.Object_Detection/dataset/test.json',
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/DINO_JIHWAN',
    type='CocoMetric')
# Added TTA in Test Pipeline
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1024, 1024), (1280, 1280), (800, 800)],
        flip=True,
        transforms=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                ),
                type='PackDetInputs'),
        ])
]

train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (320, 1333),  # 작은 스케일 추가
                                (352, 1333),
                                (384, 1333),
                                (416, 1333),
                                (448, 1333),
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (400, 4200),
                                (500, 4200),
                                (600, 4200),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(384, 600),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (320, 1333),  # 작은 스케일 추가
                                (352, 1333),
                                (384, 1333),
                                (416, 1333),
                                (448, 1333),
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (320, 1333),  # 작은 스케일 추가
                        (352, 1333),
                        (384, 1333),
                        (416, 1333),
                        (448, 1333),
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (400, 4200),
                        (500, 4200),
                        (600, 4200),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(384, 600),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (320, 1333),  # 작은 스케일 추가
                        (352, 1333),
                        (384, 1333),
                        (416, 1333),
                        (448, 1333),
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2',
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(1024, 1024), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/DINO_JIHWAN_HP_TTA_FPN_IMGSCALE'
