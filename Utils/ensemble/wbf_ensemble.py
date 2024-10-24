import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from get_mAP import calculate_mAP

def filter_by_confidence(text, conf_thresh):
    """
    Filter detection boxes based on confidence threshold.
    
    Args:
        text (str): Prediction string
        conf_thresh (float): Confidence threshold (0~1)
    
    Returns:
        str: Filtered prediction string
    """
    try:
        arr = str(text).split(' ')
        filtered_boxes = []
        
        for i in range(0, len(arr), 6):
            if i + 5 >= len(arr):
                break
            if float(arr[i+1]) > conf_thresh:
                filtered_boxes.extend(arr[i:i+6])
                
        return ' '.join(filtered_boxes)
    except Exception as e:
        print(f"Error filtering prediction string: {e}")
        return ''

def get_value(text_list, weights, iou_thr=0.3, skip_box_thr=0.0, conf_thresh=0.0):
    """
    Apply Weighted Boxes Fusion (WBF) to combine predictions from multiple models.

    Parameters:
    - text_list: List of PredictionString from each model
    - weights: List of weights for each model's predictions
    - iou_thr: IoU threshold (lower value keeps more boxes)
    - skip_box_thr: Box removal threshold (lower value keeps more boxes)
    - conf_thresh: Confidence threshold for pre-filtering boxes

    Returns:
    - fused_boxes, fused_scores, fused_labels: Combined boxes, scores, and labels
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    # Parse prediction strings into separate lists for boxes, scores, and labels
    for text in text_list:
        try:
            # First apply confidence threshold filtering
            filtered_text = filter_by_confidence(text, conf_thresh)
            arr = filtered_text.split()
            
            labels = []
            scores = []
            boxes = []
            for i in range(0, len(arr), 6):
                if i + 5 >= len(arr):
                    break
                labels.append(int(arr[i]))
                scores.append(float(arr[i+1]))
                boxes.append([
                    float(arr[i+2])/1024,  # x_min (normalized)
                    float(arr[i+3])/1024,  # y_min (normalized)
                    float(arr[i+4])/1024,  # x_max (normalized)
                    float(arr[i+5])/1024   # y_max (normalized)
                ])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        except Exception as e:
            print(f"Error parsing PredictionString: {filtered_text}. Error: {e}")
            # Handle parsing errors by adding empty lists
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
    
    # Apply Weighted Boxes Fusion
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, 
        scores_list, 
        labels_list, 
        weights=weights, 
        iou_thr=iou_thr,      
        skip_box_thr=skip_box_thr,  
        conf_type='avg'        # Use average for confidence score combination
    )
    
    return fused_boxes, fused_scores, fused_labels

def main():
    """
    Main function to process multiple prediction CSV files and create ensemble predictions.
    
    Process:
    1. Load multiple prediction CSV files
    2. Find common image IDs across all files
    3. Apply confidence threshold filtering
    4. Apply WBF to combine predictions
    5. Save results to a new CSV file
    """
    # Get number of CSV files to process
    n = int(input("Number of CSV files: "))
    dataframes = []
    
    # Load all CSV files
    for i in range(n):
        csv_path = input(f'CSV file path {i+1}: ')
        df = pd.read_csv(csv_path)
        dataframes.append(df)
    
    # Find common image IDs across all DataFrames
    common_image_ids = set(dataframes[0]['image_id'])
    for df in dataframes[1:]:
        common_image_ids &= set(df['image_id'])
    
    # Extract predictions for common image IDs
    image_ids = list(common_image_ids)
    text_lists = []
    for df in dataframes:
        df_filtered = df[df['image_id'].isin(common_image_ids)]
        df_filtered = df_filtered.set_index('image_id').reindex(image_ids)
        text_lists.append(df_filtered['PredictionString'].tolist())
    
    # Get parameters from user
    weights = list(map(float, input("Enter weights: ").split()))  # Weights for each model
    thresh = float(input("Enter the IOU threshold: "))  # IoU threshold for WBF
    conf_thresh = float(input("Enter confidence threshold (0~1): "))  # Confidence threshold

    # Process each image
    string_list = []
    for idx in range(len(image_ids)):
        # Get predictions from all models for current image
        current_texts = [text_lists[file_idx][idx] for file_idx in range(n)]
        
        # Apply WBF with confidence threshold
        boxes, scores, labels = get_value(
            current_texts, 
            weights=weights, 
            iou_thr=thresh, 
            skip_box_thr=0.00,
            conf_thresh=conf_thresh
        )
        
        # Format results back to prediction string
        string = ' '.join([
            f"{int(labels[j])} {scores[j]} " + ' '.join([f"{num*1024}" for num in boxes[j]])
            for j in range(len(labels))
        ])
        string_list.append(string)
    
    # Create final DataFrame with ensemble predictions
    final_df = pd.DataFrame({
        'PredictionString': string_list,
        'image_id': image_ids
    })
    
    # Save results
    final_df.to_csv('./submission/submission.csv', index=False)
    print("Done! CSV file created as 'submission.csv'\n")
    mean_ap, average_precisions = calculate_mAP('/data/ephemeral/home/dataset/train.json', './submission/submission.csv')
    print(f"mAP: {mean_ap}")

if __name__ == '__main__':
    main()