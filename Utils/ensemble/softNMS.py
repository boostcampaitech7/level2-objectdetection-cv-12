import pandas as pd
import numpy as np

def soft_nms(boxes, scores, sigma=0.5, iou_threshold=0.5, score_threshold=0.001):
    """
    Soft-NMS 알고리즘 적용
    :param boxes: 바운딩 박스 리스트 (x_min, y_min, x_max, y_max)
    :param scores: 바운딩 박스의 신뢰도 점수 리스트
    :param sigma: 가우시안 함수에 사용되는 표준 편차
    :param iou_threshold: 박스 겹침을 평가하는 IoU 임계값
    :param score_threshold: 제거할 박스 점수의 하한값
    :return: 최종 선택된 바운딩 박스와 해당 점수 및 클래스
    """
    N = len(boxes)
    for i in range(N):
        max_score_idx = i
        for j in range(i + 1, N):
            if scores[j] > scores[max_score_idx]:
                max_score_idx = j

        # 바운딩 박스와 점수 스왑
        boxes[i], boxes[max_score_idx] = boxes[max_score_idx], boxes[i]
        scores[i], scores[max_score_idx] = scores[max_score_idx], scores[i]

        for j in range(i + 1, N):
            iou = calculate_iou(boxes[i], boxes[j])

            if iou > iou_threshold:
                scores[j] *= np.exp(-(iou ** 2) / sigma)

        scores = [s if s >= score_threshold else 0 for s in scores]  # 점수 임계값 이하 제거

    # score가 0인 박스를 제거
    final_boxes = [boxes[i] for i in range(N) if scores[i] > 0]
    final_scores = [scores[i] for i in range(N) if scores[i] > 0]

    return final_boxes, final_scores

def calculate_iou(boxA, boxB):
    """
    IoU 계산
    :param boxA: 첫 번째 바운딩 박스
    :param boxB: 두 번째 바운딩 박스
    :return: IoU 값
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def main():
    n = int(input("num of csv files : "))
    text_lists = []
    image_ids = []

    for i in range(n):
        csv_path = input(f'CSV file path {i+1}: ')
        df = pd.read_csv(csv_path)
        if i == 0:
            image_ids = df['image_id'].tolist()
        text_lists.append(df['PredictionString'].tolist())

    thresh = float(input("iou threshold(0~1) : "))
    sigma = float(input("sigma for Soft-NMS (e.g., 0.5): "))
    score_thresh = float(input("score threshold (e.g., 0.001): "))
    print('please wait ...\n')

    string_list = []
    for idx in range(len(image_ids)):
        # 각 이미지의 바운딩 박스, 점수, 라벨 추출
        boxes = []
        scores = []
        labels = []

        for text in text_lists:
            arr = str(text[idx]).split(' ')[:-1]
            for i in range(len(arr)//6):
                labels.append(int(arr[6*i]))
                scores.append(float(arr[6*i+1]))
                boxes.append([float(arr[6*i+2]), float(arr[6*i+3]), float(arr[6*i+4]), float(arr[6*i+5])])

        # Soft-NMS 적용
        final_boxes, final_scores = soft_nms(boxes, scores, sigma=sigma, iou_threshold=thresh, score_threshold=score_thresh)
        
        string = ''
        for j in range(len(final_boxes)):
            string += str(labels[j]) + ' ' + str(final_scores[j]) + ' ' + ' '.join([str(num) for num in final_boxes[j]]) + ' '
        
        string_list.append(string.strip())

    final_df = pd.DataFrame({
        'image_id': image_ids,
        'PredictionString': string_list
    })

    final_df.to_csv('soft_nms_ensemble.csv', index=False)
    print("Done! csv file created\n")

if __name__ == '__main__':
    main()
