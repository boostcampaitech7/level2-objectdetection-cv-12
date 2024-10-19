import os

def create_image_list(data_dir, output_txt_path):
    # 폴더 안에 있는 모든 이미지 파일 경로 수집
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                # 이미지 파일의 전체 경로를 리스트에 추가
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # 경로 리스트 파일로 저장
    with open(output_txt_path, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')
    
    print(f"Image list saved to {output_txt_path}")

if __name__ == "__main__":
    # 각 폴더 경로 설정
    train_dir = "/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/train/images"
    val_dir = "/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val/images"

    # 경로 리스트 파일을 저장할 위치 설정
    train_output = "/data/ephemeral/home/level2-objectdetection-cv-12/yolov11/yolo_train.txt"
    val_output = "/data/ephemeral/home/level2-objectdetection-cv-12/yolov11/yolo_val.txt"

    # 경로 리스트 생성
    create_image_list(train_dir, train_output)
    create_image_list(val_dir, val_output)

