# configs/faster_rcnn/faster_rcnn_eva02_backbone.py

_base_ = '/data/ephemeral/home/level2-objectdetection-cv-12/baseline/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'  # 기본 Faster R-CNN 구성 파일을 참조

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='EVA02Backbone',  # 우리가 구현한 EVA02 백본
        img_size=448,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=8 / 3,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN'),
        out_indices=(7, 11, 15, 23),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/eva02_large_patch14_448.pth'  # 사전 학습된 가중치 경로
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],  # 백본의 각 출력 채널 수
        out_channels=256,
        num_outs=5
    ),
    # RPN 및 ROI 헤드 설정은 기본 Faster R-CNN 구성 파일을 사용
)

# 기타 설정은 기본 Faster R-CNN 구성 파일을 따릅니다.
