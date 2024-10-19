import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.05):
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
    
    return weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, 
        weights=weights, 
        iou_thr=iou_thr, 
        skip_box_thr=skip_box_thr
    )

def main():
    n = int(input("Number of CSV files: "))
    text_lists = [] 
    image_ids = []
    
    for i in range(n):
        csv_path = input(f'CSV file path {i+1}: ')
        df = pd.read_csv(csv_path)
        if i == 0:
            image_ids = df['image_id'].tolist()
        text_lists.append(df['PredictionString'].tolist())
    
    weights = list(map(float, input("Enter weights separated by space (e.g., 1 3): ").split()))
    thresh = float(input("Enter IoU threshold (e.g., 0.5): "))
    print('Please wait...\n')
    
    string_list = []
    for idx in range(len(image_ids)):
        current_texts = [text_lists[file_idx][idx] for file_idx in range(n)]
        boxes, scores, labels = get_value(current_texts, weights=weights, iou_thr=thresh, skip_box_thr=0.05)
        string = ' '.join([
            f"{int(labels[j])} {scores[j]:.6f} " + ' '.join([f"{num*1024:.2f}" for num in boxes[j]])
            for j in range(len(labels))
        ])
        string_list.append(string)
    
    final_df = pd.DataFrame({
        'image_id': image_ids,
        'PredictionString': string_list
    })
    
    final_df.to_csv('wbf_ensemble.csv', index=False)
    print("Done! CSV file created as 'wbf_ensemble.csv'\n")

if __name__ == '__main__':
    main()
