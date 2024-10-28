# Object Detecion Visualization
본 레포지토리는 여러 객체 탐지 모델의 시각화 구현체입니다. Train Dataset, Validation Set, Test Dataset, Ensemble 시각화를 포함하고 있습니다.

***

# 개요
- Train Dataset과 모델 예측 결과의 직관적인 시각화
- 데이터 증강을 위한 다양한 이미지 변환 기능 제공
- Confidence 기반의 예측 결과 필터링
- Ground Truth와 예측 결과의 시각적 비교 분석
- Ensemble 예측 결과의 시각적 비교 분석

***

# 파일 설명

- HOME.py: 메인 페이지 및 기본 설정
- 1_Train Data Visualization.py: Train Dataset 시각화 도구
- 2_Validation Data Visualization.py: Validation Set 예측 결과 시각화 도구
- 3_Test Data Visualization.py: Test Dataset 예측 결과 시각화 도구
- 4_Ensemble Visualization.py: Ensemble 예측 결과 시각화 도구

***

# 필요 라이브러리

```
Copystreamlit
pandas
numpy
pillow
pycocotools
albumentations
opencv-python
matplotlib
```

***

# 사용 방법
```
streamlit run HOME.py
```

## 기본 설정
```
# HOME.py 파일에서 경로 설정

dataset_path = "데이터셋 경로"
test_inf_csv_path = "테스트 데이터 추론 결과 CSV 파일 경로"
val_inf_csv_path = "검증 데이터 추론 결과 CSV 파일 경로"
val_inf_csv_path1 = "앙상블을 위한 첫 번째 검증 데이터 추론 결과 CSV 파일 경로"
val_inf_csv_path2 = "앙상블을 위한 두 번째 검증 데이터 추론 결과 CSV 파일 경로"
```

***

## 1️⃣ Train Data Visualization
![image](https://github.com/user-attachments/assets/7bcf8470-4fab-40f5-a142-7ee0a6f17279)

***

## 2️⃣ Validation Data Visualization
![image (1)](https://github.com/user-attachments/assets/f42af798-0a48-4b8b-a299-10b73481b969)

***

## 3️⃣ Test Data Visualization
![image (2)](https://github.com/user-attachments/assets/6dba88d7-dbc0-4961-b7fb-1b806e94ceb9)

***

## 4️⃣ Ensemble Visualization
![image (3)](https://github.com/user-attachments/assets/c5d3a9fd-a3c6-4835-b353-d2cd3da7217c)

***
