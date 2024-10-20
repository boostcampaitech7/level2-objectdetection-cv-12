from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO

GT_JSON = '/data/ephemeral/home/dataset/train.json'
PRED_CSV = '/data/ephemeral/home/baseline/HZ_ensemble/submission/submission.csv'
LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal",
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# load ground truth
with open(GT_JSON, 'r') as outfile:
    test_anno = (json.load(outfile))

# load prediction
pred_df = pd.read_csv(PRED_CSV)

new_pred = []

file_names = pred_df['image_id'].values.tolist()
bboxes = pred_df['PredictionString'].values.tolist()

# check variable type
for i, bbox in enumerate(bboxes):
    if isinstance(bbox, float):
        print(f'{file_names[i]} empty box')

for file_name, bbox in tqdm(zip(file_names, bboxes)):
    boxes = np.array(str(bbox).strip().split(' '))

    # boxes - class ID confidence score xmin ymin xmax ymax
    if len(boxes) % 6 == 0:
        boxes = boxes.reshape(-1, 6)
    elif isinstance(bbox, float):
        print(f'{file_name} empty box')
        continue
    else:
        raise Exception('error', 'invalid box count')
    for box in boxes:
        new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])

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
            
mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

print(mean_ap)