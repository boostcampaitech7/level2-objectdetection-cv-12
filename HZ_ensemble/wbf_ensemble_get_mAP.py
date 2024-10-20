## code source : https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master

from ensemble_boxes import *
import pandas as pd
import numpy as np
import json
from pycocotools.coco import COCO
from map_boxes import mean_average_precision_for_boxes

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001):
    boxes_list = []
    scores_list = []
    labels_list = []
    
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
    file_names = []

    n = int(input("num of csv files : "))
    text_list =[]
    
    for i in range(n):
        df = pd.read_csv(input(f'csv file path {i}: '))
        file_names = df['image_id'].values.tolist()
        text_list.append(df.PredictionString)
    
    GT_JSON = '/data/ephemeral/home/dataset/train.json'
    
    # load ground truth
    with open(GT_JSON, 'r') as outfile:
        test_anno = (json.load(outfile))

    gt = []
    coco = COCO(GT_JSON)

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        
        # new_pred에 있는 image_id만 처리
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
    weights = [0.6915, 0.7484, 0.1106]
    thresh = 0.79

    for i in range(len(text_list[0])):
        boxes, scores, labels = get_value([text[i] for text in text_list], weights= weights, iou_thr=thresh, skip_box_thr=0.00)
        for j in range(len(labels)):
            new_pred.append([file_names[i], str(int(labels[j])), str(scores[j]), boxes[j][0] * 1024, boxes[j][2] * 1024, boxes[j][1] * 1024, boxes[j][3] * 1024])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)
    print(mean_ap)


if __name__ == '__main__':
    main()