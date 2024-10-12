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
data_root = '/data/ephemeral/home/level2-objectdetection-cv-04/test_dir/2/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
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
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
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
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=10,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            110.07,117.39,123.65
        ],
        pad_size_divisor=1,
        std=[
            54.77, 53.35,54.01
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
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
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5)),
        num_layers=6),
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
        ann_file='test.json',
        data_prefix=dict(img=''),
        data_root=data_root,
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
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-04/test_dir/2/test.json',
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/Avengers_DINO',
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-04/test_dir/2/train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
        pipeline=train_pipeline,
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))


# val_cfg = dict(type='ValLoop')
# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         ann_file='k-fold-final/val_fold_1.json',
#         backend_args=None,
#         data_prefix=dict(img=''),
#         data_root=data_root,
#         metainfo=dict(
#             classes=(
#                 'General trash',
#                 'Paper',
#                 'Paper pack',
#                 'Metal',
#                 'Glass',
#                 'Plastic',
#                 'Styrofoam',
#                 'Plastic bag',
#                 'Battery',
#                 'Clothing',
#             ),
#             palette=[
#                 (
#                     220,
#                     20,
#                     60,
#                 ),
#                 (
#                     119,
#                     11,
#                     32,
#                 ),
#                 (
#                     0,
#                     0,
#                     230,
#                 ),
#                 (
#                     106,
#                     0,
#                     228,
#                 ),
#                 (
#                     60,
#                     20,
#                     220,
#                 ),
#                 (
#                     0,
#                     80,
#                     100,
#                 ),
#                 (
#                     0,
#                     0,
#                     70,
#                 ),
#                 (
#                     50,
#                     0,
#                     192,
#                 ),
#                 (
#                     250,
#                     170,
#                     30,
#                 ),
#                 (
#                     255,
#                     0,
#                     0,
#                 ),
#             ]),
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(keep_ratio=True, scale=(
#                 1024,
#                 1024,
#             ), type='Resize'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         test_mode=True,
#         type='CocoDataset'),
#     drop_last=False,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
# val_evaluator = dict(
#     ann_file='/data/ephemeral/home/level2-objectdetection-cv-04/dataset/k-fold-final/val_fold_1.json',
#     backend_args=None,
#     classwise=True,
#     format_only=False,
#     metric='bbox',
#     type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/Avengers_DINO'