
_base_ = '../configs/rtmdet/rtmdet_x_8xb32-300e_coco.py'


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

