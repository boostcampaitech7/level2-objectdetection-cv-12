# Weighted Boxes Fusion Ensemble

본 레포지토리는 여러 객체 탐지 모델의 앙상블을 위한 Weighted Boxes Fusion (WBF) 구현체입니다. mAP(mean Average Precision) 계산 및 베이지안 최적화를 통한 앙상블 가중치 최적화 도구를 포함하고 있습니다.

***

# 개요
- WBF를 사용한 다중 객체 탐지 모델의 예측 결과 통합
- 모델 평가를 위한 mAP 계산
- 앙상블 가중치와 파라미터 최적화
- 신뢰도 점수 기반의 탐지 박스 필터링

***

# 파일 설명
- wbf_ensemble.py: WBF 앙상블의 주요 구현체 (Confidence 기반 필터링 포함)
- wbf_ensemble_get_mAP.py: mAP 계산이 포함된 WBF 구현체
- get_mAP.py: 예측 결과에 대한 mAP 계산 독립 스크립트
- optimize_wbf.py: 최적의 앙상블 파라미터를 찾기 위한 베이지안 최적화

***

# 필요 라이브러리
```
pandas
numpy
pycocotools
ensemble_boxes
bayes_opt
tqdm
```

***

# 사용 방법
## 기본 앙상블
```
python wbf_ensemble.py
```

입력:

- CSV 파일 수 (모델 예측 결과)
- 예측 결과 CSV 파일 경로
- 각 모델의 가중치
- IoU 임계값
- Confidence 임계값

***

## mAP 계산
```
python get_mAP.py
```

입력:

- COCO 형식의 Ground Truth 어노테이션
- 예측 결과 CSV 파일 경로
- Confidence 임계값

***

## 앙상블 파라미터 최적화
```
python optimize_wbf.py
```

입력:

- 예측 결과 CSV 형식

```
image_id,PredictionString
1.jpg,0 0.9 100 200 150 250 1 0.8 300 400 350 450
```

최적화 파라미터:

- 모델 가중치
- IoU 임계값
- Confidence 임계값

![Snipaste_2024-10-27_00-03-23](https://github.com/user-attachments/assets/785dd7a9-4b1f-4cbe-a8f3-2adb7bfff267)
