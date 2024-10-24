import pandas as pd
import numpy as np
import json
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from map_boxes import mean_average_precision_for_boxes
from bayes_opt import BayesianOptimization

def filter_by_confidence(text, conf_thresh):
    """
    Filter detection boxes based on confidence threshold.
    
    Args:
        text (str): Prediction string
        conf_thresh (float): Confidence threshold (0~1)
    
    Returns:
        str: Filtered prediction string
    """
    arr = str(text).split(' ')[:-1]
    row = []
    for index in range(0, len(arr)-1, 6):
        if float(arr[index+1]) > conf_thresh:
            row += arr[index:index+6]
    return ' '.join(row) + ' '

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001, conf_thresh=0.0):
    """
    Performs Weighted Boxes Fusion (WBF) ensemble on predictions from multiple models
    
    Args:
        text_list (list): List of prediction strings from each model
        weights (list): Weight for each model in the ensemble
        iou_thr (float): IoU threshold for box fusion
        skip_box_thr (float): Confidence threshold for WBF
        conf_thresh (float): Confidence threshold for pre-filtering boxes
    
    Returns:
        tuple: (boxes, scores, labels) - Ensembled predictions
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for text in text_list:
        # First apply confidence threshold filtering
        filtered_text = filter_by_confidence(text, conf_thresh)
        
        # Parse filtered prediction string
        arr = str(filtered_text).split(' ')[:-1]
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

    return weighted_boxes_fusion(boxes_list, scores_list, labels_list, 
                               weights=weights, iou_thr=iou_thr, 
                               skip_box_thr=skip_box_thr)

def create_objective_function(n_files):
    """
    Creates an objective function for Bayesian optimization
    
    The objective function takes model weights, IoU threshold, and confidence threshold
    as inputs and returns mean Average Precision (mAP) as the optimization target.
    
    Args:
        n_files (int): Number of input CSV files (model predictions)
    
    Returns:
        function: Objective function that returns mAP for given parameters
    """
    def objective_function(**kwargs):
        # Extract parameters to optimize
        weights = [kwargs[f'w{i+1}'] for i in range(n_files)]
        thresh = kwargs['iou_th']
        conf_thresh = kwargs['conf_th']
        new_pred = []
        
        # Perform WBF ensemble for each image
        for i in range(min(len(df) for df in text_list)):
            try:
                boxes, scores, labels = get_value(
                    [text[i] for text in text_list], 
                    weights=weights, 
                    iou_thr=thresh,
                    skip_box_thr=0.00,
                    conf_thresh=conf_thresh
                )
                # Convert back to original scale and save predictions
                for j in range(len(labels)):
                    new_pred.append([
                        file_names[i], 
                        str(int(labels[j])), 
                        str(scores[j]), 
                        boxes[j][0] * 1024, 
                        boxes[j][2] * 1024, 
                        boxes[j][1] * 1024, 
                        boxes[j][3] * 1024
                    ])
            except Exception as e:
                print(f"Error processing index {i}: {str(e)}")
                continue

        # Format ground truth and predictions for mAP calculation
        gt_filtered = [[str(g[0]), int(g[1]), *g[2:]] for g in gt]
        pred_filtered = [[str(p[0]), int(p[1]), float(p[2]), *p[3:]] for p in new_pred]

        # Calculate mAP
        mean_ap, _ = mean_average_precision_for_boxes(
            gt_filtered, pred_filtered, iou_threshold=0.5
        )
        return mean_ap
    
    return objective_function

def main():
    """
    Main execution function
    
    Workflow:
    1. Load prediction CSV files and COCO format ground truth data
    2. Perform Bayesian optimization to find optimal weights, IoU threshold,
       and confidence threshold
    3. Output the best parameters and corresponding mAP
    """
    global file_names, text_list, gt

    file_names = []
    text_list = []

    # Load CSV files containing model predictions
    n = int(input("Number of CSV files: "))
    for i in range(n):
        df = pd.read_csv(input(f'csv file path {i + 1}: '))
        if i == 0:
            file_names = df['image_id'].values.tolist()
        text_list.append(df.PredictionString)
    
    GT_JSON = input("Enter your train.json path: ")
    
    # Load ground truth data
    with open(GT_JSON, 'r') as outfile:
        test_anno = json.load(outfile)

    gt = []
    coco = COCO(GT_JSON)

    # Extract ground truth box information
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        if file_name in file_names:
            annotation_ids = coco.getAnnIds(imgIds=image_info['id'])
            annotation_info_list = coco.loadAnns(annotation_ids)

            for annotation in annotation_info_list:
                gt.append([
                    file_name, 
                    annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    float(annotation['bbox'][1]) + float(annotation['bbox'][3])
                ])

    # Set parameter bounds for Bayesian optimization
    pbounds = {f'w{i+1}': (0.001, 1.0) for i in range(n)}
    pbounds['iou_th'] = (0.01, 1.0)  # IoU threshold
    pbounds['conf_th'] = (0.01, 1.0)  # Confidence threshold

    # Initialize and run Bayesian optimization
    optimizer = BayesianOptimization(
        f=create_objective_function(n),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,  # Number of initial random exploration points
        n_iter=2000,    # Number of optimization iterations
    )

    # Output optimization results
    print("\nBest result:")
    weights_str = ", ".join([f"{optimizer.max['params'][f'w{i+1}']:.5f}" for i in range(n)])
    print(f"Best weights: [{weights_str}]")
    print(f"Best IoU threshold: {optimizer.max['params']['iou_th']:.5f}")
    print(f"Best confidence threshold: {optimizer.max['params']['conf_th']:.5f}")
    print(f"Best mAP: {optimizer.max['target']:.5f}")

if __name__ == '__main__':
    main()