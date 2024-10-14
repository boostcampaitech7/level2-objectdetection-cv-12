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
    logger=dict(interval=300, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
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

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        act_cfg=None,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
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
param_scheduler = [
    dict(    # 1~4 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=list(range(1, 4)),
        gamma=0.82999),
    dict(    # 5 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4],
        gamma=1.748915),
    dict(    # 6 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5],
        gamma=0.120482),
    dict(    # 7 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6],
        gamma=8.3),
    dict(    # 8~10 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=list(range(7, 10)),
        gamma=0.82999),
    dict(    # 11 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10],
        gamma=1.748915),
    dict(    # 12 epoch
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.120482),
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
randomness = dict(seed=442310925)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='/data/ephemeral/home/Lv2.Object_Detection/dataset/test.json',
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2',
        metainfo=dict(
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
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    60,
                    20,
                    220,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    50,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
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
    outfile_prefix='./work_dirs/DINO_JIHWAN_AUG',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2/train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/Lv2.Object_Detection/test_dir/2',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'General trash',  # 0
                'Paper',          # 1
                'Paper pack',     # 2
                'Metal',          # 3
                'Glass',          # 4
                'Plastic',        # 5
                'Styrofoam',      # 6
                'Plastic bag',    # 7
                'Battery',        # 8
                'Clothing'        # 9
            ),
            palette=[
                (220, 20, 60),     # General trash
                (119, 11, 32),     # Paper
                (0, 0, 230),       # Paper pack
                (106, 0, 228),     # Metal
                (60, 20, 220),     # Glass
                (0, 80, 100),      # Plastic
                (0, 0, 70),        # Styrofoam
                (50, 0, 192),      # Plastic bag
                (250, 170, 30),    # Battery
                (255, 0, 0)        # Clothing
            ]
        ),
        pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomChoice',
                transforms=[
                    [
                        dict(
                            type='RandomChoiceResize',
                            keep_ratio=True,
                            scales=[
                                (480, 1333), (512, 1333), (544, 1333), (576, 1333), 
                                (608, 1333), (640, 1333), (672, 1333), (704, 1333), 
                                (736, 1333), (768, 1333), (800, 1333)
                            ]
                        )
                    ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            keep_ratio=True,
                            scales=[(400, 4200), (500, 4200), (600, 4200)]
                        ),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True
                        ),
                        dict(
                            type='RandomChoiceResize',
                            keep_ratio=True,
                            scales=[
                                (480, 1333), (512, 1333), (544, 1333), (576, 1333), 
                                (608, 1333), (640, 1333), (672, 1333), (704, 1333), 
                                (736, 1333), (768, 1333), (800, 1333)
                            ]
                        )
                    ]
                ]
            ),
            dict(type='PackDetInputs')
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 1. HorizontalFlip
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 2. Rotate
    dict(
        type='Albu',
        transforms=[
            dict(type='Rotate', limit=30, p=0.5)
        ],
    ),

    # 3. RandomBrightnessContrast
    dict(
        type='Albu',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5),
        ],
    ),
    # 4. GaussianBlur
    dict(
        type='Albu',
        transforms=[
            dict(
                type='GaussianBlur',
                blur_limit=(3, 7),
                p=0.5),
        ],
    ),
    # 5. One of {HueSaturationValue, RandomBrightnessContrast, RandomFog, RGBShift}
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0),
                    dict(
                        type='RandomFog',
                        fog_coef_lower=0.1,
                        fog_coef_upper=0.5,
                        alpha_coef=0.08,
                        p=1.0),
                    dict(
                        type='RGBShift',
                        r_shift_limit=15,
                        g_shift_limit=15,
                        b_shift_limit=15,
                        p=1.0),
                ],
                p=0.5),
        ],
    ),
    # 6. One of {Blur, MedianBlur}
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='Blur',
                        blur_limit=3,
                        p=1.0),
                    dict(
                        type='MedianBlur',
                        blur_limit=3,
                        p=1.0),
                ],
                p=0.5),
        ],
    ),
    # 7. Resize, Pad, Normalize
    dict(
        type='RandomChoiceResize',
        scales=[
            (480, 1333), (512, 1333), (544, 1333),
            (576, 1333), (608, 1333), (640, 1333),
            (672, 1333), (704, 1333), (736, 1333),
            (768, 1333), (800, 1333)
        ],
        keep_ratio=True),
    dict(type='Pad', size_divisor=32),
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
        metainfo=dict(
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
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    60,
                    20,
                    220,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    50,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
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
work_dir = './work_dirs/DINO_JIHWAN_AUG'