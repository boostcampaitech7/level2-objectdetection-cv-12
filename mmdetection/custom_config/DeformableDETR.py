#### 모델 설정 ####
# 사용할 모델 구성 파일의 상대 경로를 설정합니다.
_base_ = [
    '../configs/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco.py'
]

#### 후크 설정 ####
# 시각화를 위한 후크 설정
default_hooks = dict(
    visualization=dict(
        type="DetVisualizationHook",
        draw=True  # 시각화 그림 그리기 활성화
    )
)

# 커스텀 후크 설정
custom_hooks = [
    dict(type='SubmissionHook')  # 제출을 위한 후크
]

#### 모델 세부 설정 ####
# 클래스 수 = 10
# 손실 함수 설정: 분류는 FocalLoss, 바운딩 박스는 L1Loss, IoU는 GIoULoss
model = dict(
    bbox_head=dict(
        num_classes=10,  # 클래스 수 설정
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0  # 분류 손실 가중치
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0  # 바운딩 박스 손실 가중치
        ),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0  # IoU 손실 가중치
        )
    )
)

#### 이미지 크기 설정 ####
# 이미지 크기 = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 이미지 파일 로드
    dict(type='LoadAnnotations', with_bbox=True),  # 어노테이션 로드 (바운딩 박스 포함)
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),  # 리사이즈 (비율 유지)
    dict(type='PackDetInputs')  # 입력 데이터 패킹
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 이미지 파일 로드
    dict(type='LoadAnnotations', with_bbox=True),  # 어노테이션 로드 (바운딩 박스 포함)
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),  # 리사이즈 (비율 유지)
    dict(type='PackDetInputs')  # 입력 데이터 패킹
]

#### 학습 정책 설정 ####
# 총 에폭 수 = 12
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,      # 총 학습 에폭 수
    val_interval=1              # 검증 주기 (에폭 단위)
)
val_cfg = dict(type='ValLoop')  # 검증 루프 설정
test_cfg = dict(type='TestLoop')  # 테스트 루프 설정

# 학습률 스케줄러 설정 (MultistepLR 사용)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],  # 학습률을 감소시킬 에폭 지점
        gamma=0.1         # 학습률 감소 비율
    )
]

#### 옵티마이저 설정 ####
# 옵티마이저 = AdamW
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,          # 학습률 설정
        weight_decay=0.0001 # 가중치 감소 설정
    )
)

# 추가적인 옵티마이저 설정 (중복 정의됨)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,          # 학습률 설정
        weight_decay=0.0001 # 가중치 감소 설정
    )
)

#### 데이터셋 설정 ####
data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'

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
    batch_size=4,  # 배치 사이즈
    num_workers=5,  # 워커 수
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train.json',  # 학습 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        pipeline=train_pipeline  # 학습 파이프라인
    )
)

# 검증 데이터로더 설정
val_dataloader = dict(
    batch_size=1,  # 배치 사이즈
    num_workers=5,  # 워커 수
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        test_mode=True,  # 테스트 모드 활성화
        pipeline=test_pipeline  # 검증 파이프라인
    )
)

# 테스트 데이터로더 설정
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,  # 데이터 루트 경로
        metainfo=metainfo,    # 메타 정보 (클래스 및 색상)
        ann_file='test.json',  # 테스트 어노테이션 파일
        data_prefix=dict(img=''),  # 이미지 데이터 경로
        test_mode=True,  # 테스트 모드 활성화
        pipeline=test_pipeline  # 테스트 파이프라인
    )
)

#### 평가자 설정 ####
# 검증 평가자 설정
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val.json',  # 검증 어노테이션 파일
    metric='bbox',  # 평가 메트릭 설정 (바운딩 박스)
    format_only=False,  # 포맷만 출력하지 않음
    classwise=True,     # 클래스별 평가
)

# 테스트 평가자 설정
test_evaluator = dict(
    ann_file='/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json',  # 테스트 어노테이션 파일
)

#### WandB 설정 ####
### wandb ###
vis_backends = [
    dict(type='LocalVisBackend'),  # 로컬 시각화 백엔드
    dict(
        type='WandbVisBackend',  # WandB 시각화 백엔드
        init_kwargs={
            'project': 'Deformable_DETR',  # 프로젝트 이름 업데이트
            'entity': 'yujihwan-yonsei-university',  # 엔티티 설정
            'name': 'deformable_DETR_15EPOCH'  # 실험 이름 업데이트
        }
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'  # 시각화 도구 이름 설정
)

#### 사전 학습된 모델 로드 ####
load_from = "https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco/deformable-detr-refine_r50_16xb2-50e_coco_20221022_225303-844e0f93.pth"  # 사전 학습된 모델 경로


