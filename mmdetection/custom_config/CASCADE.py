_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../configs/_base_/default_runtime.py'
]

# 모델 설정
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # 사전 학습된 가중치 링크
num_levels = 5  # FPN의 레벨 수

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,  # 기존 백본 설정 삭제
        type='SwinTransformer',
        pretrain_img_size=384,  # 사전 학습 이미지 크기
        embed_dims=192,  # 임베딩 차원
        depths=[2, 2, 18, 2],  # 각 스테이지의 레이어 수
        num_heads=[6, 12, 24, 48],  # 각 스테이지의 헤드 수
        window_size=12,  # 윈도우 크기
        mlp_ratio=4,  # MLP 비율
        qkv_bias=True,  # QKV 편향 사용 여부
        qk_scale=None,  # QK 스케일링
        drop_rate=0.0,  # 드롭아웃 비율
        attn_drop_rate=0.0,  # 어텐션 드롭아웃 비율
        drop_path_rate=0.2,  # 드롭 패스 비율
        patch_norm=True,  # 패치 정규화 사용 여부
        out_indices=(0, 1, 2, 3),  # FPN에서 사용할 출력 인덱스
        with_cp=True,  # 체크포인트 사용 여부
        convert_weights=True,  # 가중치 변환 여부
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)  # 사전 학습된 가중치 로드
    ),
    neck=dict(
        in_channels=[192, 384, 768, 1536],  # FPN 입력 채널
        num_outs=num_levels  # FPN 출력 레벨 수
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,  # 클래스 수 설정
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9  # 분류 손실 가중치
                ),
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9
                ),
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9
                ),
            )
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,  # NMS 전에 고려할 박스 수
            max_per_img=1000,  # 이미지당 최대 박스 수
            nms=dict(type='nms', iou_threshold=0.7),  # NMS 설정
            min_bbox_size=0  # 최소 박스 크기
        ),
        rcnn=dict(
            score_thr=0.05,  # 점수 임계값
            nms=dict(type='nms', iou_threshold=0.5),  # NMS 설정
            max_per_img=300  # 이미지당 최대 박스 수
        )
    )
)

# WandB 로거 설정
### wandb ###
vis_backends = [
    dict(type='LocalVisBackend'),  # 로컬 시각화 백엔드
    dict(
        type='WandbVisBackend',  # WandB 시각화 백엔드
        init_kwargs={
            'project': 'DINO',
            'entity': 'yujihwan-yonsei-university',
            'name': 'DINO_NEWFOLD_12EPOCH'  # 예: swin-l_5scale_original_randaug_epochs6 형식으로 변경
        }
    )
]

# 로깅 및 체크포인트 설정
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200),  # 로깅 간격 설정
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,               # 체크포인트 저장 주기 (epoch 단위)
        save_best='auto',         # 자동으로 베스트 모델 저장
        max_keep_ckpts=3          # 최대 저장할 체크포인트 수
    ),
)

# 학습 스케줄 설정 (20 에폭)
max_epochs = 14
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,      # 총 학습 에폭 수
    val_interval=1              # 검증 주기 (에폭 단위)
)
val_cfg = dict(type='ValLoop')  # 검증 루프 설정
test_cfg = dict(type='TestLoop')  # 테스트 루프 설정

# 학습률 자동 조정 기본 설정
# - `enable`은 기본적으로 학습률 자동 조정 사용 여부
# - `base_batch_size` = (8 GPUs) x (2 샘플/ GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)

# AMP (자동 혼합 정밀도) 설정
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# 옵티마이저 설정 (중복 정의됨)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # DeformDETR의 경우 0.0002 사용
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),  # 그래디언트 클리핑 설정
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}  # 백본의 학습률 멀티플라이어 설정
    )
)  # custom_keys는 DeformDETR의 sampling_offsets와 reference_points를 포함함

# 커스텀 훅 설정
custom_hooks = [dict(type='SubmissionHook')]

# 학습률 스케줄러 선택 (Multistep 또는 CosineAnnealing)
param_scheduler = [
    # MultiStepLR 설정 (주석 처리됨)
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

# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'
classes = (
    "General trash", "Paper", "Paper pack", "Metal", "Glass",
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
)

backend_args = None  # 백엔드 인수 설정 (필요 시 수정)

# 학습 파이프라인 설정
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 이미지 로드
    dict(type='LoadAnnotations', with_bbox=True),               # 어노테이션 로드 (바운딩 박스 포함)
    dict(type='RandomFlip', prob=0.5),                          # 랜덤 플립 (확률 0.5)
    dict(
        type='RandomChoice',  # 랜덤 선택 트랜스폼
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',  # 랜덤 리사이즈 선택
                    scales=[
                        (1333, 480), (1333, 512), (1333, 544), (1333, 576),
                        (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                        (1333, 736), (1333, 768), (1333, 800)
                    ],
                    keep_ratio=True  # 비율 유지
                )
            ],
            [
                dict(
                    type='RandomChoiceResize',  # 랜덤 리사이즈 선택
                    # 훈련 데이터셋의 모든 이미지 비율 < 7
                    # 원래 구현을 따름
                    scales=[(4200, 400), (4200, 500), (4200, 600)],
                    keep_ratio=True
                ),
                dict(
                    type='RandomCrop',  # 랜덤 크롭
                    crop_type='absolute_range',
                    crop_size=(384, 600),  # 크롭 크기 설정
                    allow_negative_crop=True  # 음수 크롭 허용 여부
                ),
                dict(
                    type='RandomChoiceResize',  # 랜덤 리사이즈 선택
                    scales=[
                        (1333, 480), (1333, 512), (1333, 544), (1333, 576),
                        (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                        (1333, 736), (1333, 768), (1333, 800)
                    ],
                    keep_ratio=True  # 비율 유지
                )
            ]
        ]
    ),
    dict(
        type='PackDetInputs',  # 입력 데이터 패킹
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 검증 파이프라인 설정
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 이미지 로드
    dict(
        type='Resize',
        scale=(1333, 800),       # 리사이즈 크기 설정
        keep_ratio=True,         # 비율 유지
        backend='pillow'         # 백엔드 설정
    ),
    # GT 어노테이션이 없으면 파이프라인 삭제
    dict(type='LoadAnnotations', with_bbox=True),  # 어노테이션 로드 (바운딩 박스 포함)
    dict(
        type='PackDetInputs',  # 입력 데이터 패킹
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 테스트 파이프라인 설정
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 이미지 로드
    dict(
        type='Resize',
        scale=(1333, 800),       # 리사이즈 크기 설정
        keep_ratio=True,         # 비율 유지
        backend='pillow'         # 백엔드 설정
    ),
    # GT 어노테이션이 없으면 파이프라인 삭제
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',  # 입력 데이터 패킹
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 학습 데이터로더 설정
train_dataloader = dict(
    batch_size=8,  # 배치 사이즈
    num_workers=4,  # 워커 수
    persistent_workers=True,  # 워커 지속 여부
    sampler=dict(type='DefaultSampler', shuffle=True),  # 샘플러 설정 (셔플링 활성화)
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 배치 샘플러 설정
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',  # 학습 어노테이션 파일
        data_prefix=dict(img=data_root),  # 이미지 데이터 경로
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 필터링 설정 (빈 GT 제거, 최소 크기)
        pipeline=train_pipeline,  # 학습 파이프라인
        metainfo=dict(classes=classes),  # 메타 정보 (클래스 이름)
        backend_args=backend_args  # 백엔드 인수
    )
)

# 검증 데이터로더 설정
val_dataloader = dict(
    batch_size=4,  # 배치 사이즈
    num_workers=4,  # 워커 수
    persistent_workers=True,  # 워커 지속 여부
    drop_last=False,  # 마지막 배치를 드롭하지 않음
    sampler=dict(type='DefaultSampler', shuffle=False),  # 샘플러 설정 (셔플링 비활성화)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
        data_prefix=dict(img=data_root),  # 이미지 데이터 경로
        test_mode=True,  # 테스트 모드 활성화
        pipeline=test_pipeline,  # 검증 파이프라인
        metainfo=dict(classes=classes),  # 메타 정보 (클래스 이름)
        backend_args=backend_args  # 백엔드 인수
    )
)

# 테스트 데이터로더 설정
test_dataloader = dict(
    batch_size=1,  # 배치 사이즈
    num_workers=8,  # 워커 수
    persistent_workers=True,  # 워커 지속 여부
    drop_last=False,  # 마지막 배치를 드롭하지 않음
    sampler=dict(type='DefaultSampler', shuffle=False),  # 샘플러 설정 (셔플링 비활성화)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',  # 테스트 어노테이션 파일
        data_prefix=dict(img=data_root),  # 이미지 데이터 경로
        test_mode=True,  # 테스트 모드 활성화
        pipeline=test_pipeline,  # 테스트 파이프라인
        metainfo=dict(classes=classes),  # 메타 정보 (클래스 이름)
        backend_args=backend_args  # 백엔드 인수
    )
)

# 검증 평가자 설정
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
    metric='bbox',  # 평가 메트릭 설정 (바운딩 박스)
    format_only=False,  # 포맷만 출력하지 않음
    classwise=True,  # 클래스별 평가
    backend_args=backend_args  # 백엔드 인수
)

# 테스트 평가자 설정
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',  # 테스트 어노테이션 파일
    metric='bbox',  # 평가 메트릭 설정 (바운딩 박스)
    format_only=False,  # 포맷만 출력하지 않음
    classwise=True,  # 클래스별 평가
    backend_args=backend_args  # 백엔드 인수
)
