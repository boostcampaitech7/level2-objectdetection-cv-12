_base_ = [
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

### Model ###
model = dict(
    type='ConditionalDETR',  # Changed from 'DINO' to 'ConditionalDETR'
    num_queries=300,  # Adjusted number of queries as per CO-DETR requirements
    with_box_refine=False,
    as_two_stage=False,  # CO-DETR typically uses a single-stage approach
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    num_feature_levels=4,  # CO-DETR might use fewer feature levels
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
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),  # Adjusted to match num_feature_levels
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,  # Updated num_levels
                               dropout=0.1),  # Increased dropout for CO-DETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1))),  # Increased dropout for CO-DETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.1),  # Increased dropout
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,  # Updated num_levels
                                dropout=0.1),  # Increased dropout
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1)),  # Increased dropout
        post_norm_cfg=dict(type='LN')),  # Added layer normalization
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='ConditionalDETRHead',  # Changed from 'DINOHead' to 'ConditionalDETRHead'
        num_classes=10,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # Remove dn_cfg as it's specific to DINO
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

#### Hooks ####
# hook for visualization
# default_hooks = dict(visualization=dict(type="DetVisualizationHook",draw=True))

# custom hooks
custom_hooks = [dict(type='SubmissionHook')]

### Dataset
# Augmentation
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

### Augmentation Rule 확인 !!!
image_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(  # LSJ
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    # dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

### JSON 파일 정보 확인 !!!
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))

### Evaluation ###
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='CocoMetric',  # Ensure consistency
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

#### Learning Policy ####
max_epochs = 15

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Select Multistep or CosineAnnealing
param_scheduler = [
    # dict(
    #     type='MultiStepLR',
    #     begin=1,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[10, 14],  # Adjusted milestones for CO-DETR
    #     gamma=0.1),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

### Optimizer ###
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # Adjust learning rate if necessary
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

### wandb ###
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'ConditionalDETR',  # Updated project name
            'entity': 'yujihwan-yonsei-university',
            'name': 'CO_DETR_NEWFOLD_15EPOCH'  # Updated experiment name
         })]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'