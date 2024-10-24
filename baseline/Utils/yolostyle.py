import pandas as pd

# CSV 파일 경로
file_path = '/data/ephemeral/home/level2-objectdetection-cv-12/yolov11/yolov8_project/experiment_14/test_results_yolo.csv'

# CSV 파일 불러오기
df = pd.read_csv(file_path)

# 'image_id' 열에서 'images/'를 'test/'로 변경
df['image_id'] = df['image_id'].str.replace('images/', 'test/')

# 변경된 내용을 새 CSV 파일로 저장
updated_file_path = '/data/ephemeral/home/level2-objectdetection-cv-12/yolov11/updated_test_results_yolo.csv'
df.to_csv(updated_file_path, index=False)
