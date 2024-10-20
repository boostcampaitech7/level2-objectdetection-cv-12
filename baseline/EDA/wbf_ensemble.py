import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

def get_value(text_list, weights, iou_thr=0.3, skip_box_thr=0.0):
    """
    WBF를 적용하여 박스를 결합하는 함수.

    Parameters:
    - text_list: 각 모델의 PredictionString 리스트
    - weights: 각 모델의 가중치 리스트
    - iou_thr: IoU 임계값 (낮출수록 더 많은 박스 유지)
    - skip_box_thr: 박스 제거 임계값 (낮추면 더 많은 박스 유지)

    Returns:
    - fused_boxes, fused_scores, fused_labels: 결합된 박스, 점수, 레이블
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for text in text_list:
        try:
            arr = str(text).split(' ')
            labels = []
            scores = []
            boxes = []
            for i in range(0, len(arr), 6):
                if i + 5 >= len(arr):
                    break
                labels.append(int(arr[i]))
                scores.append(float(arr[i+1]))
                boxes.append([
                    float(arr[i+2])/1024,  # x_min
                    float(arr[i+3])/1024,  # y_min
                    float(arr[i+4])/1024,  # x_max
                    float(arr[i+5])/1024   # y_max
                ])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        except Exception as e:
            print(f"Error parsing PredictionString: {text}. Error: {e}")
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
    
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, 
        scores_list, 
        labels_list, 
        weights=weights, 
        iou_thr=iou_thr,       # IoU 임계값을 낮춤
        skip_box_thr=skip_box_thr,  # 박스 제거 임계값을 낮춤
        conf_type='avg'        # 점수 결합 방식을 평균으로 설정
    )
    
    return fused_boxes, fused_scores, fused_labels

def main():
    """
    메인 함수: 사용자로부터 CSV 파일을 입력받아 WBF를 적용하고 결과를 CSV로 저장.
    """
    n = int(input("Number of CSV files: "))
    text_lists = [] 
    image_ids = []
    
    for i in range(n):
        csv_path = input(f'CSV file path {i+1}: ')
        df = pd.read_csv(csv_path)
        if i == 0:
            image_ids = df['image_id'].tolist()
        text_lists.append(df['PredictionString'].tolist())
    
    # 모든 모델에 동일한 가중치를 부여
    weights = [1.0] * n
    print(f"Using weights: {weights}")
    
    # IoU 임계값과 박스 제거 임계값을 낮춰서 최대한 많은 박스를 유지
    iou_thr = 0.3  # 기본값 0.5에서 낮춤
    skip_box_thr = 0.0  # 기본값 0.05에서 낮춤
    print(f"Using IoU threshold: {iou_thr} and skip box threshold: {skip_box_thr}")
    
    print('Please wait...\n')
    
    string_list = []
    for idx in range(len(image_ids)):
        current_texts = [text_lists[file_idx][idx] for file_idx in range(n)]
        boxes, scores, labels = get_value(
            current_texts, 
            weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr
        )
        # 박스 좌표를 원래 스케일로 되돌림 (1024 배)
        string = ' '.join([
            f"{int(labels[j])} {scores[j]:.6f} " + ' '.join([f"{num*1024:.2f}" for num in boxes[j]])
            for j in range(len(labels))
        ])
        string_list.append(string)
    
    final_df = pd.DataFrame({
        'image_id': image_ids,
        'PredictionString': string_list
    })
    
    final_df.to_csv('wbf_ensemble_max_boxes.csv', index=False)
    print("Done! CSV file created as 'wbf_ensemble_max_boxes.csv'\n")

if __name__ == '__main__':
    main()

