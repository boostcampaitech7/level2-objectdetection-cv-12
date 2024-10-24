_base_ = [
    '../configs/_base_/datasets/coco_detection.py',  # COCO 데이터셋 기본 설정
    '../configs/_base_/default_runtime.py'           # 기본 런타임 설정
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # 사전 학습된 Swin Transformer 가중치 링크

### 모델 설정 ###
model = dict(
    type='DINO',  # 모델 타입 설정 (DINO)
    num_queries=900,  # 매칭 쿼리 수 설정
    with_box_refine=True,  # 박스 정제 사용 여부
    as_two_stage=True,  # 두 단계 방식 사용 여부
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # 이미지 평균값
        std=[58.395, 57.12, 57.375],     # 이미지 표준편차
        bgr_to_rgb=True,                 # BGR을 RGB로 변환
        pad_size_divisor=1                # 패딩 크기 배수
    ),
    num_feature_levels=5,  # 특징 레벨 수 설정
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,  # 사전 학습 이미지 크기
        embed_dims=192,         # 임베딩 차원
        depths=[2, 2, 18, 2],   # 각 스테이지의 레이어 수
        num_heads=[6, 12, 24, 48],  # 각 스테이지의 헤드 수
        window_size=12,         # 윈도우 크기
        mlp_ratio=4,            # MLP 비율
        qkv_bias=True,          # QKV 편향 사용 여부
        qk_scale=None,          # QK 스케일링
        drop_rate=0.0,          # 드롭아웃 비율
        attn_drop_rate=0.0,     # 어텐션 드롭아웃 비율
        drop_path_rate=0.2,     # 드롭 패스 비율
        patch_norm=True,        # 패치 정규화 사용 여부
        out_indices=(0, 1, 2, 3),  # FPN에서 사용할 출력 인덱스
        with_cp=True,           # 체크포인트 사용 여부
        convert_weights=True,   # 가중치 변환 여부
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)  # 사전 학습된 가중치 로드
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],  # 입력 채널 수 설정
        kernel_size=1,                      # 커널 크기
        out_channels=256,                   # 출력 채널 수
        act_cfg=None,                       # 활성화 함수 설정
        norm_cfg=dict(type='GN', num_groups=32),  # 정규화 설정 (Group Normalization)
        num_outs=5                          # 출력 레벨 수
    ),
    encoder=dict(
        num_layers=6,  # 인코더 레이어 수
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, 
                num_levels=5,
                dropout=0.0  # 드롭아웃 비율 (DeformDETR의 경우 0.1)
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 피드포워드 채널 수 (DeformDETR의 경우 1024)
                ffn_drop=0.0  # 드롭아웃 비율 (DeformDETR의 경우 0.1)
            )
        )
    ),
    decoder=dict(
        num_layers=6,  # 디코더 레이어 수
        return_intermediate=True,  # 중간 결과 반환 여부
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, 
                num_heads=8,
                dropout=0.0  # 드롭아웃 비율 (DeformDETR의 경우 0.1)
            ),
            cross_attn_cfg=dict(
                embed_dims=256, 
                num_levels=5,
                dropout=0.0  # 드롭아웃 비율 (DeformDETR의 경우 0.1)
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 피드포워드 채널 수 (DeformDETR의 경우 1024)
                ffn_drop=0.0  # 드롭아웃 비율 (DeformDETR의 경우 0.1)
            )
        ),
        post_norm_cfg=None  # 후처리 정규화 설정
    ),
    positional_encoding=dict(
        num_feats=128,    # 피처 수
        normalize=True,   # 정규화 여부
        offset=0.0,       # 오프셋 (DeformDETR의 경우 -0.5)
        temperature=20    # 온도 파라미터 (DeformDETR의 경우 10000)
    ),
    bbox_head=dict(
        type='DINOHead',
        num_classes=10,  # 클래스 수 설정
        sync_cls_avg_factor=True,  # 클래스 평균 동기화 여부
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0  # 분류 손실 가중치 (DeformDETR의 경우 2.0)
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0  # 바운딩 박스 손실 가중치
        ),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0  # IoU 손실 가중치
        )
    ),
    dn_cfg=dict(  # DN 설정 (TODO: 모델의 train_cfg로 이동할 것)
        label_noise_scale=0.5,    # 라벨 노이즈 스케일
        box_noise_scale=1.0,      # 박스 노이즈 스케일 (DN-DETR의 경우 0.4)
        group_cfg=dict(
            dynamic=True,
            num_groups=None,
            num_dn_queries=100  # DN 쿼리 수 (절반으로 설정할 것)
        )
    ),
    # 학습 및 테스트 설정
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),  # FocalLoss 비용
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),  # BBox L1 비용
                dict(type='IoUCost', iou_mode='giou', weight=2.0)  # IoU 비용
            ]
        )
    ),
    test_cfg=dict(
        max_per_img=300  # 이미지당 최대 박스 수 (DeformDETR의 경우 100)
    )
)

#### 후크 설정 ####
# 시각화를 위한 후크 설정 (주석 처리됨)
# default_hooks = dict(visualization=dict(type="DetVisualizationHook",draw=True))

# 커스텀 후크 설정
custom_hooks = [
    dict(type='SubmissionHook')  # 제출을 위한 후크
]

### 데이터셋 설정 ###
# 데이터 증강 설정
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

# 증강 규칙 확인 !!!
image_size = (1024, 1024)  # 이미지 크기 설정

# 학습 파이프라인 설정
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),  # 이미지 파일 로드
    dict(type='LoadAnnotations', with_bbox=True),  # 어노테이션 로드 (바운딩 박스 포함)
    dict(type='Resize', scale=image_size, keep_ratio=True),  # 리사이즈 (비율 유지)
    dict(type='RandomFlip', prob=0.5),  # 랜덤 플립 (확률 0.5)
    dict(  # LSJ 증강
        type='RandomResize',
        scale=image_size,            # 리사이즈 크기
        ratio_range=(0.1, 2.0),      # 비율 범위
        keep_ratio=True              # 비율 유지
    ),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,        # 크롭 크기 설정
        recompute_bbox=True,         # 바운딩 박스 재계산
        allow_negative_crop=True     # 음수 크롭 허용 여부
    ),
    # 랜덤 증강 주석 처리됨
    # dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='PackDetInputs')  # 입력 데이터 패킹
]

# 테스트 파이프라인 설정
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 이미지 파일 로드
    dict(type='Resize', scale=image_size, keep_ratio=True),  # 리사이즈 (비율 유지)
    # GT 어노테이션이 없으면 파이프라인 삭제
    dict(type='LoadAnnotations', with_bbox=True),  # 어노테이션 로드 (바운딩 박스 포함)
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')  # 메타 키 설정
    )
]

data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'  # 데이터 루트 경로

metainfo = {
    'classes': (
        'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',
    ),
    'palette': [  # 클래스별 색상 설정
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

# 학습 데이터로더 설정
train_dataloader = dict(
    batch_size=2,  # 배치 사이즈
    num_workers=4,  # 워커 수
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',  # 학습 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        pipeline=train_pipeline    # 학습 파이프라인
    )
)

# 검증 데이터로더 설정
val_dataloader = dict(
    batch_size=1,  # 배치 사이즈
    num_workers=4,  # 워커 수
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        pipeline=test_pipeline,    # 검증 파이프라인
        test_mode=True             # 테스트 모드 활성화
    )
)

# 테스트 데이터로더 설정
test_dataloader = dict(
    batch_size=8,  # 배치 사이즈
    num_workers=4,  # 워커 수
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',  # 테스트 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        pipeline=test_pipeline,    # 테스트 파이프라인
        test_mode=True             # 테스트 모드 활성화
    )
)

### 평가자 설정 ###
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
    metric='bbox',          # 평가 메트릭 설정 (바운딩 박스)
    format_only=False,      # 포맷만 출력하지 않음
    classwise=True,         # 클래스별 평가
)

test_evaluator = dict(
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json'  # 테스트 어노테이션 파일
)

#### 학습 정책 설정 ####
max_epochs = 15  # 총 에폭 수

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,      # 총 학습 에폭 수
    val_interval=1              # 검증 주기 (에폭 단위)
)
val_cfg = dict(type='ValLoop')  # 검증 루프 설정
test_cfg = dict(type='TestLoop')  # 테스트 루프 설정

# 학습률 스케줄러 선택 (Multistep 또는 CosineAnnealing)
param_scheduler = [
    # MultiStepLR 스케줄러 설정 (주석 처리됨)
    # dict(
    #     type='MultiStepLR',
    #     begin=1,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[4, 5],
    #     gamma=0.1
    # ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,                # 최소 학습률
        begin=0,                    
        T_max=max_epochs,           # 코사인 주기 설정
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True  # 이터레이션 기반으로 변환
    )
]

### 옵티마이저 설정 ###
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,          # 학습률 설정 (DeformDETR의 경우 0.0002)
        weight_decay=0.0001 # 가중치 감소 설정
    ),
    clip_grad=dict(
        max_norm=0.1,       # 그래디언트 클리핑 최대 노름
        norm_type=2         # 노름 타입 (L2 노름)
    ),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}  # 백본의 학습률 멀티플라이어 설정
    )
)  # custom_keys는 DeformDETR의 sampling_offsets와 reference_points를 포함함

### WandB 시각화 설정 ###
vis_backends = [
    dict(type='LocalVisBackend'),  # 로컬 시각화 백엔드
    dict(
        type='WandbVisBackend',      # WandB 시각화 백엔드
        init_kwargs={
            'project': 'DINO',        # 프로젝트 이름 설정
            'entity': 'yujihwan-yonsei-university',  # 엔티티 설정
            'name': 'DINO_NEWFOLD_12EPOCH'  # 실험 이름 설정 (예: swin-l_5scale_original_randaug_epochs6 형식으로 변경)
        }
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'  # 시각화 도구 이름 설정
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)  # 학습률 자동 조정 설정

#### 사전 학습된 모델 로드 ####
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'  # 사전 학습된 모델 경로
