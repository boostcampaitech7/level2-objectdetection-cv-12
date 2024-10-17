# 필요한 라이브러리 임포트
import os
import json
import mmengine
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

# 데이터셋에서 레이블과 그룹 추출
def extract_labels_and_groups(data):
    labels = []
    groups = []
    for img in data['images']:
        img_annotations = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]
        if img_annotations:
            # 첫 번째 어노테이션의 카테고리 ID를 레이블로 사용
            labels.append(img_annotations[0]['category_id'])
        else:
            labels.append(0)  # 어노테이션이 없는 경우 0으로 설정
        groups.append(img['id'])
    return np.array(labels), np.array(groups)

# 어노테이션 파일 저장 함수
def save_anns(name, images, annotations, original_data, out_dir):
    sub_anns = {
        'images': images,
        'annotations': annotations,
        'licenses': original_data['licenses'],
        'categories': original_data['categories'],
        'info': original_data['info']
    }
    mmengine.mkdir_or_exist(out_dir)
    mmengine.dump(sub_anns, os.path.join(out_dir, name))

# 이미지의 file_name 업데이트 함수
def update_file_names(images, prefix):
    for img in images:
        img['file_name'] = os.path.join(prefix, os.path.basename(img['file_name']))
    return images

# Stratified Group K-Fold 분할 함수
def stratified_group_kfold_split(data, out_dir, fold):
    labels, groups = extract_labels_and_groups(data)
    sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=2024)
    for f, (train_idx, val_idx) in enumerate(sgkf.split(X=groups, y=labels, groups=groups), 1):
        # 인덱스를 사용하여 이미지 리스트 분할
        train_images = [data['images'][i] for i in train_idx]
        val_images = [data['images'][i] for i in val_idx]

        # 인덱스를 이미지 ID로 변환
        train_image_ids = [data['images'][i]['id'] for i in train_idx]
        val_image_ids = [data['images'][i]['id'] for i in val_idx]

        # 이미지 ID를 사용하여 어노테이션 분할
        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]

        # file_name 업데이트
        train_images = update_file_names(train_images, 'train')
        val_images = update_file_names(val_images, 'val')

        # 어노테이션 파일 저장
        save_anns(f'train_fold_{f}.json', train_images, train_annotations, data, out_dir)
        save_anns(f'val_fold_{f}.json', val_images, val_annotations, data, out_dir)

# 데이터 로드 및 K-Fold 분할 실행
def main():
    data_root = '/data/ephemeral/home/level2-objectdetection-cv-12/dataset'
    out_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/k-fold-final'
    fold = 10

    with open(os.path.join(data_root, 'train.json'), 'r') as f:
        data = json.load(f)

    stratified_group_kfold_split(data, out_dir, fold)

if __name__ == '__main__':
    main()
