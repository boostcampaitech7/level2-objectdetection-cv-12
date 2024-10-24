from ensemble_boxes import *
import pandas as pd
import numpy as np
import json
from pycocotools.coco import COCO
from map_boxes import mean_average_precision_for_boxes

def filter_by_confidence(df, thresh):
    """
    Filter detection boxes based on confidence threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing predictions
        thresh (float): Confidence threshold (0~1)
    
    Returns:
        pd.DataFrame: DataFrame with filtered predictions
    """
    string_list = []
    for string in df.PredictionString:
        arr = list(map(str, str(string).split(' ')))
        row = []
        for index in range(0, len(arr)-1, 6):
            if float(arr[index+1]) > thresh:
                row += arr[index:index+6]
        string_list.append(' '.join(row) + ' ')
    
    df = df.copy()
    df.PredictionString = string_list
    return df

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Process prediction strings and perform Weighted Boxes Fusion (WBF).
    
    Args:
        text_list (list): List of prediction strings from different models
        weights (list): Weight values for each model's predictions
        iou_thr (float): IoU threshold for box fusion
        skip_box_thr (float): Threshold to filter out boxes with low confidence
    
    Returns:
        tuple: Fused boxes, confidence scores, and class labels
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    # Parse prediction strings into separate lists for boxes, scores, and labels
    for text in text_list:
        arr = str(text).split(' ')[:-1]
        labels = []
        scores = []
        boxes = []
        for i in range(len(arr)//6):
            labels.append(int(arr[6*i]))
            scores.append(float(arr[6*i+1]))
            boxes.append([float(i)/1024 for i in arr[6*i+2:6*i+6]])
        
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0001)

def main():
    """
    Main function to perform WBF ensemble and calculate mAP.
    
    Process:
    1. Load multiple CSV files containing model predictions
    2. Filter predictions based on confidence threshold
    3. Process ground truth annotations from COCO format JSON
    4. Perform WBF ensemble on filtered predictions
    5. Calculate mAP for the ensemble results
    """
    file_names = []
    n = int(input("Number of CSV files: "))
    dataframes = []

    # Load prediction CSV files
    for i in range(n):
        csv_path = input(f'CSV file path {i+1}: ')
        df = pd.read_csv(csv_path)
        dataframes.append(df)
    
    # Get confidence threshold and filter predictions
    conf_thresh = float(input("Enter confidence threshold (0~1): "))
    dataframes = [filter_by_confidence(df, conf_thresh) for df in dataframes]
    
    # Ensure all DataFrames have matching indices
    common_index = dataframes[0].index.intersection(dataframes[1].index)
    for df in dataframes[2:]:
        common_index = common_index.intersection(df.index)
    
    for df in dataframes:
        df = df.loc[common_index]
    
    file_names = dataframes[0]['image_id'].values.tolist()

    GT_JSON = input("Enter your train.json path: ")
    
    # Load and process ground truth annotations from COCO format
    with open(GT_JSON, 'r') as outfile:
        test_anno = json.load(outfile)

    gt = []
    coco = COCO(GT_JSON)

    # Extract ground truth boxes for evaluation
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        if file_name in file_names:
            annotation_ids = coco.getAnnIds(imgIds=image_info['id'])
            annotation_info_list = coco.loadAnns(annotation_ids)

            for annotation in annotation_info_list:
                gt.append([file_name, annotation['category_id'],
                        float(annotation['bbox'][0]),
                        float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                        float(annotation['bbox'][1]),
                        float(annotation['bbox'][1]) + float(annotation['bbox'][3])])

    new_pred = []
    weights = list(map(float, input("Enter weights: ").split()))  # Weights for each model
    thresh = float(input("Enter the IOU threshold: "))  # IoU threshold for WBF

    # Perform WBF ensemble on predictions
    for row in zip(*[df.itertuples() for df in dataframes]):
        text_list = [row[i].PredictionString for i in range(n)]
        boxes, scores, labels = get_value(
            text_list, 
            weights=weights, 
            iou_thr=thresh, 
            skip_box_thr=0.00
        )
        for j in range(len(labels)):
            new_pred.append([row[0].image_id, str(int(labels[j])), str(scores[j]), 
                           boxes[j][0] * 1024, boxes[j][2] * 1024, 
                           boxes[j][1] * 1024, boxes[j][3] * 1024])

    # Calculate and print mAP
    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)
    print(f"mAP: {mean_ap}")


if __name__ == '__main__':
    main()