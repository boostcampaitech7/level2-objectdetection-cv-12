from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO

def filter_predictions_by_confidence(pred_df, confidence_threshold):
    """
    Filter prediction boxes based on confidence threshold.
    
    Args:
    pred_df (pd.DataFrame): DataFrame containing predictions
    confidence_threshold (float): Minimum confidence threshold (0-1)
    
    Returns:
    pd.DataFrame: Filtered DataFrame with predictions above threshold
    """
    string_list = []
    for pred_string in pred_df.PredictionString:
        if isinstance(pred_string, float):  # Handle NaN cases
            string_list.append('')
            continue
            
        boxes = str(pred_string).strip().split(' ')
        filtered_boxes = []
        
        # Process boxes in groups of 6 values
        for i in range(0, len(boxes)-1, 6):
            if float(boxes[i+1]) > confidence_threshold:
                filtered_boxes.extend(boxes[i:i+6])
                
        string_list.append(' '.join(filtered_boxes) + ' ' if filtered_boxes else '')
    
    filtered_df = pred_df.copy()
    filtered_df['PredictionString'] = string_list
    return filtered_df

def calculate_mAP(gt_json_path, pred_csv_path, confidence_threshold=0.0):
    """
    Calculate mean Average Precision (mAP) by comparing prediction results with ground truth annotations.
    
    Args:
    gt_json_path (str): Path to the ground truth JSON file in COCO format
    pred_csv_path (str): Path to the prediction CSV file
    confidence_threshold (float): Minimum confidence threshold for predictions (0-1)
    
    Returns:
    float: The calculated mAP value
    dict: A dictionary of average precisions for each class
    """
    
    # Load ground truth annotations from COCO format JSON
    with open(gt_json_path, 'r') as outfile:
        test_anno = json.load(outfile)

    # Load and filter predictions based on confidence threshold
    pred_df = pd.read_csv(pred_csv_path)
    filtered_pred_df = filter_predictions_by_confidence(pred_df, confidence_threshold)

    new_pred = []  # List to store processed predictions

    # Extract image IDs and filtered bounding box predictions
    file_names = filtered_pred_df['image_id'].values.tolist()
    bboxes = filtered_pred_df['PredictionString'].values.tolist()

    # Check for empty predictions
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float) or bbox.strip() == '':
            print(f'{file_names[i]} empty box')

    # Process predictions into required format
    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        if isinstance(bbox, float) or bbox.strip() == '':
            continue
            
        boxes = np.array(str(bbox).strip().split(' '))

        # Validate box format
        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        
        # Convert box coordinates and append to new_pred list
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])

    # Process ground truth annotations
    gt = []
    coco = COCO(gt_json_path)

    # Iterate through all images in the ground truth
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
                
    # Calculate mAP using IoU threshold of 0.5
    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

    return mean_ap, average_precisions

# Class labels for reference
LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal",
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

if __name__ == "__main__":
    GT_JSON = input("Enter your train.json path: ")
    PRED_CSV = input("Enter your prediction csv path: ")
    CONFIDENCE_THRESHOLD = float(input("Enter confidence threshold (0-1): "))
    
    mean_ap, average_precisions = calculate_mAP(GT_JSON, PRED_CSV, CONFIDENCE_THRESHOLD)
    print(f"\nmAP: {mean_ap}")