import json

# Load the original COCO dataset
with open('/data/ephemeral/home/Lv2.Object_Detection/level2-objectdetection-cv-12/dataset/train.json', 'r') as f:
    coco_data = json.load(f)

# Set the maximum number of boxes allowed per image
max_boxes = 40

# Create a dictionary to store the count of annotations per image
annotation_counts = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotation_counts:
        annotation_counts[img_id] = 0
    annotation_counts[img_id] += 1

# Filter images and annotations
filtered_images = [img for img in coco_data['images'] if annotation_counts.get(img['id'], 0) <= max_boxes]
filtered_image_ids = set(img['id'] for img in filtered_images)
filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_image_ids]

# Create the new COCO dataset with filtered images and annotations
filtered_coco_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': coco_data['categories']
}

# Save the filtered COCO dataset to a new JSON file
output_path = '/data/ephemeral/home/Lv2.Object_Detection/level2-objectdetection-cv-12/dataset/filtered_train.json'
with open(output_path, 'w') as f:
    json.dump(filtered_coco_data, f, indent=4)

print(f'Filtered dataset saved with a max of {max_boxes} boxes per image at: {output_path}')
