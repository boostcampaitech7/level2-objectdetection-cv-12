"""
Dataset Splitting Script using StratifiedGroupKFold

이 스크립트는 COCO 형식의 JSON 어노테이션 파일을 사용하여 데이터를 5개의 폴드로 분할하고,
각 폴드에 대해 학습 및 검증 데이터를 생성합니다. 또한, 분할된 데이터셋을 새로운 디렉토리에 복사합니다.

사용 방법:
1. `annotation` 변수에 원본 `train.json` 파일의 경로를 설정합니다.
2. 스크립트를 실행하고, CSV 파일의 수, 각 CSV 파일의 경로, 가중치, IoU 임계값을 입력합니다.
3. 분할된 데이터셋이 지정된 디렉토리에 저장됩니다.

필요 라이브러리:
- pandas
- numpy
- sklearn
- shutil
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import os
import shutil
import sys

def load_annotations(annotation_path):
    """
    JSON 어노테이션 파일을 로드합니다.

    Args:
        annotation_path (str): JSON 어노테이션 파일의 경로.

    Returns:
        dict: 로드된 JSON 데이터.
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_kfold(data, n_splits=5, random_state=411):
    """
    StratifiedGroupKFold을 사용하여 데이터셋을 분할합니다.

    Args:
        data (dict): COCO 형식의 JSON 데이터.
        n_splits (int): 폴드의 수.
        random_state (int): 랜덤 시드.

    Returns:
        StratifiedGroupKFold: 설정된 KFold 객체.
        np.ndarray: 타겟 레이블 배열.
        np.ndarray: 그룹 배열.
    """
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']), 1))  # 단순한 특성 행렬
    y = np.array([v[1] for v in var])  # 카테고리 ID
    groups = np.array([v[0] for v in var])  # 이미지 ID

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cv, y, groups

def print_fold_distributions(cv, X, y, groups):
    """
    각 폴드의 학습 및 검증 데이터의 레이블 분포를 출력합니다.

    Args:
        cv (StratifiedGroupKFold): KFold 객체.
        X (np.ndarray): 특성 행렬.
        y (np.ndarray): 타겟 레이블.
        groups (np.ndarray): 그룹 배열.
    """
    print("KFold 분할 결과:")
    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"폴드 {fold_ind + 1}:")
        print("  학습 그룹:", groups[train_idx])
        print("  학습 레이블:", y[train_idx])
        print("  검증 그룹:", groups[val_idx])
        print("  검증 레이블:", y[val_idx])
        print("-" * 50)

def get_distribution(y):
    """
    레이블의 분포를 백분율로 계산합니다.

    Args:
        y (np.ndarray): 레이블 배열.

    Returns:
        list of str: 각 클래스의 분포 비율 (예: ['50.00%', '30.00%', ...]).
    """
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) + 1)]

def create_distribution_dataframe(cv, X, y, groups, data, n_splits=5):
    """
    각 폴드의 학습 및 검증 데이터 분포를 데이터프레임으로 생성합니다.

    Args:
        cv (StratifiedGroupKFold): KFold 객체.
        X (np.ndarray): 특성 행렬.
        y (np.ndarray): 타겟 레이블.
        groups (np.ndarray): 그룹 배열.
        data (dict): COCO 형식의 JSON 데이터.
        n_splits (int): 폴드의 수.

    Returns:
        pd.DataFrame: 각 폴드의 분포를 나타내는 데이터프레임.
    """
    distrs = [get_distribution(y)]
    index = ['전체 데이터']
    categories = [cat['name'] for cat in data['categories']]

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_y, val_y = y[train_idx], y[val_idx]
        train_gr, val_gr = groups[train_idx], groups[val_idx]

        # 학습과 검증 그룹 간 중복 확인
        assert len(set(train_gr) & set(val_gr)) == 0, "학습과 검증 그룹이 겹칩니다."

        distrs.append(get_distribution(train_y))
        distrs.append(get_distribution(val_y))
        index.append(f'학습 - 폴드{fold_ind + 1}')
        index.append(f'검증 - 폴드{fold_ind + 1}')

    df = pd.DataFrame(distrs, index=index, columns=categories)
    return df

def split_and_copy_dataset(data, cv, new_dataset_dir, origin_dataset_dir, fold_ind, val_ratio=0.2):
    """
    데이터셋을 학습 및 검증으로 분할하고, 이미지를 새로운 디렉토리에 복사합니다.

    Args:
        data (dict): COCO 형식의 JSON 데이터.
        cv (StratifiedGroupKFold): KFold 객체.
        new_dataset_dir (str): 새로운 데이터셋 디렉토리 경로.
        origin_dataset_dir (str): 원본 데이터셋 디렉토리 경로.
        fold_ind (int): 현재 폴드 인덱스.
        val_ratio (float): 검증 데이터 비율.
    """
    # 각 폴드에 대한 분할 수행
    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(np.ones(len(data['annotations'])), y, groups)):
        print(f"폴드 {fold_ind + 1} 처리 중...")
        
        # 학습 및 검증 그룹 추출
        train_gr, val_gr = groups[train_idx], groups[val_idx]
        image_ids_train, image_ids_val = set(train_gr), set(val_gr)
        num_train, num_val = len(image_ids_train), len(image_ids_val)
        
        # 이미지와 어노테이션 분할
        train_images = [img for img in data['images'] if img['id'] in image_ids_train]
        val_images = [img for img in data['images'] if img['id'] in image_ids_val]
        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids_train]
        val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids_val]
        
        # 검증 이미지 파일 이름 수정 (val 폴더로 이동)
        for img_info in val_images:
            name = os.path.basename(img_info['file_name'])
            img_info['file_name'] = os.path.join('val', name)
        
        # 새로운 JSON 데이터 생성
        train_data = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': data['categories'],
        }

        val_data = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': data['categories'],
        }

        # 새로운 폴드 디렉토리 생성
        fold_dir = os.path.join(new_dataset_dir, str(fold_ind))
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'test'), exist_ok=True)  # 테스트 데이터 폴더

        # 새로운 JSON 파일 경로 설정
        new_train_json = os.path.join(fold_dir, 'train.json')
        new_val_json = os.path.join(fold_dir, 'val.json')
        copy_test_json = os.path.join(fold_dir, 'test.json')

        # 새로운 JSON 파일 저장
        with open(new_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer)

        with open(new_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer)

        # 원본 데이터에서 테스트 JSON 복사
        shutil.copyfile(os.path.join(origin_dataset_dir, 'test.json'), copy_test_json)

        # 학습 이미지 복사
        for train_img in train_images:
            src_path = os.path.join(origin_dataset_dir, train_img['file_name'])
            dest_path = os.path.join(fold_dir, 'train', os.path.basename(train_img['file_name']))
            shutil.copyfile(src_path, dest_path)

        # 검증 이미지 복사
        for val_img in val_images:
            src_path = os.path.join(origin_dataset_dir, 'train', os.path.basename(val_img['file_name']))
            dest_path = os.path.join(fold_dir, 'val', os.path.basename(val_img['file_name']))
            shutil.copyfile(src_path, dest_path)

        # 테스트 이미지 전체 복사
        shutil.copytree(os.path.join(origin_dataset_dir, 'test'), os.path.join(fold_dir, 'test'), dirs_exist_ok=True)

        # 결과 출력
        print(f'폴드 {fold_ind + 1} 완료:')
        print(f'  학습 이미지 개수 ({int((1 - val_ratio) * 100)}%): {num_train}')
        print(f'  새 학습 디렉토리 파일 개수: {len(os.listdir(os.path.join(fold_dir, "train")))}')
        print(f'  검증 이미지 개수 ({int(val_ratio * 100)}%): {num_val}')
        print(f'  새 검증 디렉토리 파일 개수: {len(os.listdir(os.path.join(fold_dir, "val")))}\n')

def main():
    """
    메인 함수:
    - JSON 어노테이션 파일을 로드합니다.
    - StratifiedGroupKFold을 사용하여 데이터셋을 분할합니다.
    - 각 폴드에 대해 학습 및 검증 데이터를 생성하고, 이미지를 새로운 디렉토리에 복사합니다.
    """
    try:
        # 원본 어노테이션 파일 경로 설정
        annotation_path = '/data/ephemeral/home/level2-objectdetection-cv-12/dataset/train.json'
        origin_dataset_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/dataset'
        new_dataset_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD'
        input_json_path = annotation_path
        val_ratio = 0.2  # 검증 데이터 비율

        # JSON 어노테이션 파일 로드
        data = load_annotations(annotation_path)
        print("어노테이션 파일 로드 완료.")

        # KFold 준비
        cv, y, groups = prepare_kfold(data, n_splits=5, random_state=411)
        print("StratifiedGroupKFold 준비 완료.")

        # 폴드 분할 및 분포 출력
        print_fold_distributions(cv, np.ones((len(y), 1)), y, groups)
        df_distribution = create_distribution_dataframe(cv, np.ones((len(y), 1)), y, groups, data, n_splits=5)
        print("\n레이블 분포 데이터프레임:")
        print(df_distribution)

        # 데이터셋 분할 및 복사
        split_and_copy_dataset(data, cv, new_dataset_dir, origin_dataset_dir, fold_ind=0, val_ratio=val_ratio)
        print("모든 폴드에 대한 데이터셋 분할 및 복사 완료.")

    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
