import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
import matplotlib.font_manager as fm
from PIL import ImageFont
from ensemble_boxes import *
from functools import lru_cache
import os, sys
from HOME import dataset_path, val_inf_csv_path1, val_inf_csv_path2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
st.set_page_config(layout="wide")


def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001):
    boxes_list = []
    scores_list = []
    labels_list = []

    for text in text_list:
        arr = str(text).split(" ")[:-1]
        labels = []
        scores = []
        boxes = []
        for i in range(len(arr) // 6):
            labels.append(int(arr[6 * i]))
            scores.append(float(arr[6 * i + 1]))
            boxes.append([float(i) / 1024 for i in arr[6 * i + 2 : 6 * i + 6]])

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.0001,
    )


@st.cache_data
def get_new_pred_anno_df(csv_path1, csv_path2, weights, iou_thr):
    dataframes = []

    df1 = pd.read_csv(csv_path1)
    dataframes.append(df1)
    df2 = pd.read_csv(csv_path2)
    dataframes.append(df2)

    common_image_ids = set(dataframes[0]["image_id"])
    for df in dataframes[1:]:
        common_image_ids &= set(df["image_id"])

    image_ids = list(common_image_ids)
    text_lists = []
    for df in dataframes:
        df_filtered = df[df["image_id"].isin(common_image_ids)]
        df_filtered = df_filtered.set_index("image_id").reindex(image_ids)
        text_lists.append(df_filtered["PredictionString"].tolist())

    weights = weights
    iou_thr = iou_thr

    skip_box_thr = 0.00

    string_list = []
    for idx in range(len(image_ids)):
        current_texts = [text_lists[file_idx][idx] for file_idx in range(2)]
        boxes, scores, labels = get_value(
            current_texts, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        string = " ".join(
            [
                f"{int(labels[j])} {scores[j]} "
                + " ".join([f"{num*1024}" for num in boxes[j]])
                for j in range(len(labels))
            ]
        )
        string_list.append(string)

    final_df = pd.DataFrame({"PredictionString": string_list, "image_id": image_ids})

    return image_ids, final_df


@st.cache_data
def load_csv(csv_path, classes):
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


@st.cache_data
def load_df(df, classes):
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
    def __init__(self, folder_path, csv_path1, csv_path2):
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
        self.csv_path1 = csv_path1
        self.csv_path2 = csv_path2
        self.pred1_anno_df = load_csv(csv_path1, self.classes)
        self.pred2_anno_df = load_csv(csv_path2, self.classes)

    @lru_cache(maxsize=None)
    def get_font(self, size=25):
        font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        return ImageFont.truetype(font_path, size)

    def draw_image(self, image, annotations):
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
            st.write("")

            # 새로운 슬라이더 추가
            st.subheader("Ensemble Settings")
            weight_1 = st.slider("Weight 1", 0.0, 1.0, 0.5, 0.05)
            weight_2 = st.slider("Weight 2", 0.0, 1.0, 0.5, 0.05)
            iou_thr = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

            # 가중치 정규화
            total_weight = weight_1 + weight_2
            normalized_weight1 = weight_1 / total_weight
            normalized_weight2 = weight_2 / total_weight

            # 새로운 가중치와 IOU 임계값으로 ensemble 결과 업데이트
            exist_file_names, pred3_anno_df = get_new_pred_anno_df(
                self.csv_path1,
                self.csv_path2,
                [normalized_weight1, normalized_weight2],
                iou_thr,
            )

            self.pred3_anno_df = load_df(pred3_anno_df, self.classes)
            self.exist_file_names = exist_file_names

        if filename:
            while len(filename) < 4:
                filename = "0" + filename

            pred1_filtered_df = self.pred1_anno_df[
                self.pred1_anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)
            pred1_filtered_df = self.filter_annotations(
                pred1_filtered_df, confidence_threshold
            )

            pred2_filtered_df = self.pred2_anno_df[
                self.pred2_anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)
            pred2_filtered_df = self.filter_annotations(
                pred2_filtered_df, confidence_threshold
            )

            pred3_filtered_df = self.pred3_anno_df[
                self.pred3_anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)
            pred3_filtered_df = self.filter_annotations(
                pred3_filtered_df, confidence_threshold
            )

            pred1_image_path = f"{self.folder_path}/train/{filename}.jpg"
            pred2_image_path = f"{self.folder_path}/train/{filename}.jpg"
            pred3_image_path = f"{self.folder_path}/train/{filename}.jpg"

            if not (
                os.path.exists(pred1_image_path)
                or os.path.exists(pred2_image_path)
                or os.path.exists(pred3_image_path)
            ):
                st.error(f"File {filename} does not exist.")
                return

            if not (("train/" + filename + ".jpg") in self.exist_file_names):
                st.error(f"File {filename} does not exist.")
                return

            pred1_original_image = self._load_image(pred1_image_path)
            pred2_original_image = self._load_image(pred2_image_path)
            pred3_original_image = self._load_image(pred3_image_path)

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("Pred1")
                if not pred1_filtered_df.empty:
                    pred1_final_image = self.draw_image(
                        pred1_original_image, pred1_filtered_df
                    )
                    st.image(pred1_final_image, use_column_width=True)
                else:
                    st.image(pred1_original_image)

            with col2:
                st.subheader("Pred2")
                if not pred2_filtered_df.empty:
                    pred2_final_image = self.draw_image(
                        pred2_original_image, pred2_filtered_df
                    )
                    st.image(pred2_final_image, use_column_width=True)
                else:
                    st.image(pred2_original_image)

            with col3:
                st.subheader("Ensembled")
                if not pred3_filtered_df.empty:
                    pred3_final_image = self.draw_image(
                        pred3_original_image, pred3_filtered_df
                    )
                    st.image(pred3_final_image, use_column_width=True)
                else:
                    st.image(pred3_original_image)


if __name__ == "__main__":
    folder_path = dataset_path
    csv_path1 = val_inf_csv_path1
    csv_path2 = val_inf_csv_path2

    viewer = DataViewer(folder_path, csv_path1, csv_path2)
    viewer.run()
