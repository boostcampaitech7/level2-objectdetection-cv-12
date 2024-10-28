import os
import json

def convert_coco_to_yolo(coco_json_path, images_dir, labels_dir):
    # json 파일 열기
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 이미지 ID별 이미지 정보 저장
    image_info_dict = {img['id']: img for img in coco_data['images']}
    
    # 어노테이션을 이미지 ID별로 정리
    yolo_info_dict = {}
    for anno in coco_data['annotations']:
        image_id = anno['image_id']
        img_info = image_info_dict[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]

        # 클래스 ID와 바운딩 박스 정보 추출
        class_id = anno['category_id']
        bbox = anno['bbox']
        xmin, ymin, bbox_width, bbox_height = bbox
        center_x = xmin + bbox_width / 2
        center_y = ymin + bbox_height / 2

        # YOLO 형식으로 변환 (정규화)
        center_x /= img_width
        center_y /= img_height
        bbox_width /= img_width
        bbox_height /= img_height

        # 해당 이미지 파일의 텍스트 정보 추가
        if file_name not in yolo_info_dict:
            yolo_info_dict[file_name] = []
        yolo_info_dict[file_name].append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # YOLO 형식의 라벨 파일 생성
    os.makedirs(labels_dir, exist_ok=True)
    for file_name, lines in yolo_info_dict.items():
        label_file_path = os.path.join(labels_dir, f"{file_name}.txt")
        with open(label_file_path, 'w') as f:
            f.write("\n".join(lines))
    print(f"YOLO 라벨 파일 생성 완료: {labels_dir}")

if __name__ == "__main__":
    # 경로 설정
    coco_json_path = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test.json'  # COCO 형식 JSON 파일 경로
    images_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/test'          # 이미지 파일이 있는 디렉토리 경로
    labels_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/yolov11/yolo_test_label'          # YOLO 라벨 파일을 저장할 디렉토리 경로

    # 변환 실행
    convert_coco_to_yolo(coco_json_path, images_dir, labels_dir)
