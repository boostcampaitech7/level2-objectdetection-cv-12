#### model config ####
# 사용할 model config 주소 상대 주소로 넣기
_base_ = [
    '../configs/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco.py'
]



#### hooks ####
# hook for visualization
default_hooks = dict(visualization=dict(type="DetVisualizationHook",draw=True))

# custom hooks
custom_hooks = [dict(type='SubmissionHook')]



#### model ####
# num_classes = 10
# loss -> cls = FocalLoss, bbox = L1Loss, iou = GIoULoss
model = dict(
    bbox_head=dict(
        num_classes=10,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
        ))



#### img size ####
# img size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]



#### learning policy ####
# epochs = 12
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]



#### optimizer ####
# optimizer = AdamW
# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
#     )
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001))


#### dataset ####
data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline))



#### evaluation ####
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    )

test_evaluator = dict(ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json')



### wandb ###
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'Deformable_DETR',  # Updated project name
            'entity': 'yujihwan-yonsei-university',
            'name': 'deformable_DETR_15EPOCH'  # Updated experiment name
         })]


visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')



#### pretrained model ####
load_from = "https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco/deformable-detr-refine_r50_16xb2-50e_coco_20221022_225303-844e0f93.pth" 

