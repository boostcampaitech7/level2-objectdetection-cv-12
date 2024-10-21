import csv

def augment_bounding_boxes(input_csv, output_csv, delta=5):
    """
    COCO 형식의 CSV 파일을 읽고 바운딩 박스를 조금씩 이동시켜 복제합니다.
    복제된 바운딩 박스는 원본과 함께 출력 CSV 파일에 저장됩니다.

    매개변수:
    - input_csv: 입력 CSV 파일 경로
    - output_csv: 생성된 데이터를 저장할 출력 CSV 파일 경로
    - delta: 바운딩 박스를 이동시킬 픽셀 수
    """
    # 이동할 방향 설정: 왼쪽, 오른쪽, 위쪽, 아래쪽
    shifts = [
        {'xmin_shift': -delta, 'ymin_shift': 0, 'xmax_shift': -delta, 'ymax_shift': 0},  # 왼쪽으로 이동
        {'xmin_shift': delta, 'ymin_shift': 0, 'xmax_shift': delta, 'ymax_shift': 0},    # 오른쪽으로 이동
        {'xmin_shift': 0, 'ymin_shift': -delta, 'xmax_shift': 0, 'ymax_shift': -delta},  # 위쪽으로 이동
        {'xmin_shift': 0, 'ymin_shift': delta, 'xmax_shift': 0, 'ymax_shift': delta},    # 아래쪽으로 이동
    ]

    with open(input_csv, 'r') as f_in, open(output_csv, 'w') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            if not row:
                continue

            # PredictionString 및 이미지 경로 분리
            prediction_str, image_path = ','.join(row[:-1]), row[-1]

            # 원본 데이터 기록
            writer.writerow([prediction_str, image_path])

            # PredictionString 파싱
            tokens = prediction_str.strip().split()
            if len(tokens) != 6:
                print(f"Invalid line: {row}")
                continue

            label, score = tokens[0], tokens[1]
            xmin, ymin, xmax, ymax = map(float, tokens[2:6])

            for shift in shifts:
                new_xmin = xmin + shift['xmin_shift']
                new_ymin = ymin + shift['ymin_shift']
                new_xmax = xmax + shift['xmax_shift']
                new_ymax = ymax + shift['ymax_shift']

                # 새로운 바운딩 박스가 유효한지 확인
                if new_xmin < 0 or new_ymin < 0 or new_xmin >= new_xmax or new_ymin >= new_ymax:
                    continue

                new_prediction_str = f"{label} {score} {new_xmin} {new_ymin} {new_xmax} {new_ymax}"
                # 복제된 데이터 기록
                writer.writerow([new_prediction_str, image_path])

if __name__ == "__main__":
    input_csv = '/data/ephemeral/home/level2-objectdetection-cv-12/DINO_NEWFOLD.csv'         # 입력 CSV 파일 경로
    output_csv = '/data/ephemeral/home/level2-objectdetection-cv-12/output_aug.csv'   # 출력 CSV 파일 경로
    delta = 5                       # 이동할 픽셀 수

    augment_bounding_boxes(input_csv, output_csv, delta)
