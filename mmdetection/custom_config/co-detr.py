_base_ = '/data/ephemeral/home/baseline/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py'

data_root = '/data/ephemeral/home/FOLD/'

image_size = (1024, 1024)
dataset_type = 'CocoDataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

backend_args = None

num_classes = len(classes)

model = dict(
    type='CoDETR',
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=12.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=num_classes,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_dn_queries=500, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=num_classes,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=6),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=False),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=120.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=12.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=num_classes,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
)


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
                    scales=[(240, 1024), (256, 1024), (272, 1024), (288, 1024),
                            (304, 1024), (320, 1024), (336, 1024), (352, 1024),
                            (368, 1024), (384, 1024), (400, 1024), (416, 1024),
                            (432, 1024), (448, 1024), (464, 1024), (480, 1024),
                            (496, 1024), (512, 1024), (528, 1024), (544, 1024),
                            (560, 1024), (576, 1024), (592, 1024), (608, 1024),
                            (624, 1024), (640, 1024), (656, 1024), (672, 1024),
                            (688, 1024), (704, 1024), (720, 1024), (736, 1024),
                            (752, 1024)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(200, 2100), (250, 2100), (300, 2100)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(192, 300),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(240, 1024), (256, 1024), (272, 1024), (288, 1024),
                            (304, 1024), (320, 1024), (336, 1024), (352, 1024),
                            (368, 1024), (384, 1024), (400, 1024), (416, 1024),
                            (432, 1024), (448, 1024), (464, 1024), (480, 1024),
                            (496, 1024), (512, 1024), (528, 1024), (544, 1024),
                            (560, 1024), (576, 1024), (592, 1024), (608, 1024),
                            (624, 1024), (640, 1024), (656, 1024), (672, 1024),
                            (688, 1024), (704, 1024), (720, 1024), (736, 1024),
                            (752, 1024)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='/data/ephemeral/home/FOLD/',
        ann_file='train.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img='./'),
        pipeline=train_pipeline,
        backend_args=None
    ),
    collate_fn=dict(type='pseudo_collate')
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type='CocoDataset',
        test_mode=True,
        data_root='/data/ephemeral/home/FOLD/',
        ann_file='train.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img='./'),
        pipeline=test_pipeline,
        backend_args=None,
    ),
    collate_fn=dict(type='pseudo_collate')
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type='CocoDataset',
        test_mode=True,
        data_root='/data/ephemeral/home/FOLD/',
        ann_file='test.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img='./'),
        pipeline=test_pipeline,
        backend_args=None,
    ),
    collate_fn=dict(type='pseudo_collate')
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file ='/data/ephemeral/home/FOLD/train.json',
    metric=['bbox']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/FOLD/test.json',
    metric=['bbox'],
    format_only=True,
    outfile_prefix='./work_dirs/coco_detection/test'
)