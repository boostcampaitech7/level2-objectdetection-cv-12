_base_ = '../configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py'

model = dict(bbox_head=dict(num_classes=10))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'LAST_ATSS_JIHWAN',
            'entity': 'yujihwan-yonsei-university',
            'name': 'MIMIC_ATSS_JIHWAN'
         })
]
visualizer = dict(type='DetLocalVisualizer',vis_backends=vis_backends,name='visualizer')


default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=1,save_best='auto',max_keep_ckpts=2))