import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
import matplotlib.font_manager as fm
from PIL import ImageFont
from functools import lru_cache
import os, sys
from HOME import dataset_path, val_inf_csv_path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
st.set_page_config(layout="wide")


@st.cache_data
def load_csv(csv_path, classes):
    """Load and parse prediction CSV file"""
    df = pd.read_csv(csv_path)
    annotation = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        pred_string = row["PredictionString"]

        if pd.isna(pred_string):
            continue

        preds = pred_string.split()

        for i in range(0, len(preds), 6):
            if i + 5 < len(preds):
                confidence = float(preds[i + 1])
                class_id = int(float(preds[i]))
                x_min = float(preds[i + 2])
                y_min = float(preds[i + 3])
                x_max = float(preds[i + 4])
                y_max = float(preds[i + 5])

                annotation.append(
                    {
                        "image_id": image_id,
                        "class_id": class_id,
                        "class_name": classes[class_id],
                        "confidence": confidence,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "activate": True,
                    }
                )

    return pd.DataFrame(annotation)


class DataViewer:
    def __init__(self, folder_path, csv_path):
        self.folder_path = folder_path
        self.classes = [
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]
        self.palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 165, 0),
            (27, 27, 27),
            (0, 255, 255),
            (255, 0, 255),
            (165, 42, 42),
            (255, 192, 203),
        ]
        self.gt_anno_df = self.json2df(self.folder_path, self.classes)
        self.pred_anno_df = load_csv(csv_path, self.classes)
        self.exist_file_names = self.pred_anno_df["image_id"].values.tolist()

    @staticmethod
    @st.cache_data
    def json2df(folder_path, classes):
        """Convert COCO JSON annotations to DataFrame"""
        coco = COCO(f"{folder_path}/train.json")
        train_df = pd.DataFrame()
        image_ids, class_name, class_id = [], [], []
        x_min, y_min, x_max, y_max = [], [], [], []

        for image_id in coco.getImgIds():
            image_info = coco.loadImgs(image_id)[0]
            ann_ids = coco.getAnnIds(imgIds=image_info["id"])
            anns = coco.loadAnns(ann_ids)
            file_name = image_info["file_name"]

            for ann in anns:
                image_ids.append(file_name)
                class_name.append(classes[ann["category_id"]])
                class_id.append(ann["category_id"])
                x_min.append(float(ann["bbox"][0]))
                y_min.append(float(ann["bbox"][1]))
                x_max.append(float(ann["bbox"][0]) + float(ann["bbox"][2]))
                y_max.append(float(ann["bbox"][1]) + float(ann["bbox"][3]))

        train_df["image_id"] = image_ids
        train_df["class_name"] = class_name
        train_df["class_id"] = class_id
        train_df["x_min"] = x_min
        train_df["y_min"] = y_min
        train_df["x_max"] = x_max
        train_df["y_max"] = y_max
        train_df["activate"] = True  # Add 'activate' column with default value True

        return train_df

    @lru_cache(maxsize=None)
    def get_font(self, size=25):
        font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        return ImageFont.truetype(font_path, size)

    def draw_image(self, image, annotations, dataset_type):
        draw = ImageDraw.Draw(image)
        font = self.get_font(size=25)
        width, height = image.size

        for idx, row in annotations.iterrows():
            if row["activate"]:
                x_min = max(0, min(row["x_min"], width - 1))
                y_min = max(0, min(row["y_min"], height - 1))
                x_max = max(0, min(row["x_max"], width - 1))
                y_max = max(0, min(row["y_max"], height - 1))

                if x_max > x_min and y_max > y_min:
                    box = [(x_min, y_min), (x_max, y_max)]
                    color = self.palette[int(row["class_id"])]
                    draw.rectangle(box, outline=color, width=3)
                    if dataset_type == "gt":
                        draw.text(
                            (x_min, y_min - 30),
                            f"{idx} {row['class_name']}",
                            fill=color,
                            font=font,
                        )
                    else:
                        draw.text(
                            (x_min, y_min - 30),
                            f"{idx} {row['class_name']} {row['confidence']:.2f}",
                            fill=color,
                            font=font,
                        )

        return image

    @staticmethod
    @st.cache_data
    def _load_image(path):
        return Image.open(path)

    def filter_annotations(self, df, confidence_threshold):
        return df[df["confidence"] >= confidence_threshold].reset_index(drop=True)

    def run(self):
        with st.sidebar:
            st.header("Menu")
            filename = st.text_input("File name")
            confidence_threshold = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
            )

        if filename:
            while len(filename) < 4:
                filename = "0" + filename

            gt_filtered_df = self.gt_anno_df[
                self.gt_anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)

            pred_filtered_df = self.pred_anno_df[
                self.pred_anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)
            pred_filtered_df = self.filter_annotations(
                pred_filtered_df, confidence_threshold
            )

            gt_image_path = f"{self.folder_path}/train/{filename}.jpg"
            pred_image_path = f"{self.folder_path}/train/{filename}.jpg"

            if not (os.path.exists(gt_image_path) or os.path.exists(pred_image_path)):
                st.error(f"File {filename} does not exist.")
                return

            if not (("train/" + filename + ".jpg") in self.exist_file_names):
                st.error(f"File {filename} does not exist.")
                return

            gt_original_image = self._load_image(gt_image_path)
            pred_original_image = self._load_image(pred_image_path)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Ground Truth")
                if not gt_filtered_df.empty:
                    gt_final_image = self.draw_image(
                        gt_original_image, gt_filtered_df, "gt"
                    )
                    st.image(gt_final_image, use_column_width=True)
                else:
                    st.write(
                        "No bounding boxes found for the given confidence threshold."
                    )

            with col2:
                st.subheader("Prediction")
                if not pred_filtered_df.empty:
                    pred_final_image = self.draw_image(
                        pred_original_image, pred_filtered_df, "pred"
                    )
                    st.image(pred_final_image, use_column_width=True)
                else:
                    st.image(pred_original_image)


if __name__ == "__main__":
    folder_path = dataset_path
    csv_path = val_inf_csv_path
    viewer = DataViewer(folder_path, csv_path)
    viewer.run()
