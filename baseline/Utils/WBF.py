# 코드 출처: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master

from ensemble_boxes import weighted_boxes_fusion  # 앙상블 박스 융합을 위한 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001):
    """
    여러 예측 문자열을 처리하여 박스, 점수, 레이블 목록을 생성하고
    Weighted Boxes Fusion을 적용하여 최종 박스, 점수, 레이블을 반환합니다.

    Args:
        text_list (list of str): 예측 문자열 목록.
        weights (list of int): 각 예측의 가중치.
        iou_thr (float): IoU 임계값.
        skip_box_thr (float): 박스 스킵 임계값.

    Returns:
        tuple: 최종 박스, 점수, 레이블 목록.
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for text in text_list:
        # 예측 문자열을 공백으로 분리하고 마지막 요소는 제외
        arr = str(text).split(' ')[:-1]
        labels = []
        scores = []
        boxes = []
        
        # 6개 단위로 레이블, 점수, 박스 좌표 추출
        for i in range(len(arr) // 6):
            labels.append(int(arr[6 * i]))
            scores.append(float(arr[6 * i + 1]))
            # 박스 좌표를 1024로 나누어 정규화
            boxes.append([float(coord) / 1024 for coord in arr[6 * i + 2:6 * i + 6]])
        
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Weighted Boxes Fusion 적용
    return weighted_boxes_fusion(
        boxes_list, 
        scores_list, 
        labels_list, 
        weights=weights, 
        iou_thr=iou_thr, 
        skip_box_thr=skip_box_thr
    )

def main():
    """
    메인 함수:
    여러 CSV 파일에서 예측을 읽어들여 Weighted Boxes Fusion을 적용하고
    최종 예측을 새로운 CSV 파일로 저장합니다.
    """
    try:
        # 사용자로부터 CSV 파일 수 입력 받기
        n = int(input("CSV 파일 수 입력: "))
        text_list = [] 
        
        # 각 CSV 파일에서 PredictionString 컬럼 읽어오기
        for i in range(n):
            csv_path = input(f'CSV 파일 경로 {i + 1}: ')
            df = pd.read_csv(csv_path)
            text_list.append(df.PredictionString)
    
        # 가중치 입력 받기
        weights = list(map(int, input("가중치 입력 (예: 1 3): ").split()))
        # IoU 임계값 입력 받기
        thresh = float(input("IoU 임계값 (0~1): "))
        print('잠시만 기다려 주세요...\n')
    
        string_list = []
        # 각 예측에 대해 WBF 적용
        for i in range(len(text_list[0])):
            # 모든 CSV 파일의 i번째 예측을 가져와 WBF 적용
            boxes, scores, labels = get_value(
                [text[i] for text in text_list], 
                weights=weights, 
                iou_thr=thresh, 
                skip_box_thr=0.0001
            )
            string = ''
            # WBF 결과를 문자열 형식으로 변환
            for j in range(len(labels)):
                string += f"{int(labels[j])} {scores[j]} " + ' '.join([f"{num * 1024:.2f}" for num in boxes[j]]) + ' '
            string_list.append(string.strip())
    
        # 마지막 CSV 파일의 DataFrame에 WBF 결과 저장
        df['PredictionString'] = string_list
        # 결과를 새로운 CSV 파일로 저장
        df.to_csv('wbf_ensemble.csv', index=False)
        print("완료! 'wbf_ensemble.csv' 파일이 생성되었습니다.\n")
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == '__main__':
    main()
