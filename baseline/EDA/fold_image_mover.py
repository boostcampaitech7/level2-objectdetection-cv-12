import json
import os
import shutil

# JSON 파일 경로
json_path = '/data/ephemeral/home/level2-objectdetection-cv-12/k-fold-final/val_fold.json'

# 원본 이미지가 있는 경로 (이 경로에서 이미지를 찾습니다)
src_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/dataset/train'

# 이미지를 복사할 대상 경로 (이 경로로 이미지를 복사합니다)
dest_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/selected_images'

# JSON 파일을 읽어오기
with open(json_path, 'r') as f:
    data = json.load(f)

# 주석이 있는 이미지를 추출하여 복사
copied_images = 0
for image_info in data['images']:
    # 이미지 ID를 기준으로 주석이 있는지 확인
    image_id = image_info['id']
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    
    if annotations:  # 주석이 있는 이미지
        file_name = image_info['file_name']
        
        # file_name에 절대 경로가 포함되지 않도록 조정
        src_path = os.path.join(src_dir, os.path.basename(file_name))
        dest_path = os.path.join(dest_dir, os.path.basename(file_name))
        
        # 대상 경로에 이미지 저장 폴더 생성
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 이미지 파일 복사
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            copied_images += 1
        else:
            print(f"Warning: {src_path} 파일이 존재하지 않습니다.")

print(f"{copied_images}개의 이미지가 {dest_dir}로 복사되었습니다.")
