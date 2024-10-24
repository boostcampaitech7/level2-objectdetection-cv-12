import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.font_manager as fm
from PIL import ImageFont
import albumentations as A
import cv2
from functools import lru_cache
import os, sys
from HOME import dataset_path, test_inf_csv_path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
st.set_page_config(layout="wide")


@st.cache_data
def load_csv(csv_file_path, classes):
    df = pd.read_csv(csv_file_path)
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
        self.anno_df = load_csv(csv_path, self.classes)

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

    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        brightness = int(brightness * 255 / 100)
        contrast = float(contrast * 0.01)

        image = cv2.convertScaleAbs(image, alpha=contrast + 1, beta=brightness)
        return image

    def apply_gaussian_blur(self, image, blur):
        return cv2.GaussianBlur(image, (blur, blur), 0)

    def apply_transform(
        self, image, annotations, transform, brightness, contrast, blur
    ):
        image = np.array(image)
        image = self.adjust_brightness_contrast(image, brightness, contrast)

        if blur > 0:
            image = self.apply_gaussian_blur(image, blur)

        bboxes = annotations[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        labels = annotations["class_id"].tolist()

        transformed = transform(image=image, bboxes=bboxes, labels=labels)

        new_bboxes = pd.DataFrame(
            transformed["bboxes"], columns=["x_min", "y_min", "x_max", "y_max"]
        )
        new_labels = pd.Series(transformed["labels"], name="class_id")

        valid_boxes = (new_bboxes["x_max"] > new_bboxes["x_min"]) & (
            new_bboxes["y_max"] > new_bboxes["y_min"]
        )
        new_bboxes = new_bboxes[valid_boxes]
        new_labels = new_labels[valid_boxes]

        annotations = annotations.iloc[: len(new_bboxes)].copy()
        annotations[["x_min", "y_min", "x_max", "y_max"]] = new_bboxes
        annotations["class_id"] = new_labels
        annotations["class_name"] = annotations["class_id"].map(
            lambda x: self.classes[int(x)]
        )

        return Image.fromarray(transformed["image"]), annotations

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
                st.write("Transform")

                transform_list = []

                if st.checkbox("HorizontalFlip"):
                    transform_list.append(A.HorizontalFlip(p=1))
                if st.checkbox("VerticalFlip"):
                    transform_list.append(A.VerticalFlip(p=1))

                if st.checkbox("Rotate"):
                    angle = st.slider(
                        "Rotation Angle", min_value=-30, max_value=30, value=0, step=1
                    )
                    transform_list.append(
                        A.Rotate(limit=(angle, angle), border_mode=2, value=0, p=1)
                    )

                # Brightness and Contrast
                if st.sidebar.checkbox("Brightness & Contrast"):
                    brightness = st.slider(
                        "Brightness", min_value=-100, max_value=100, value=0, step=5
                    )
                    contrast = st.slider(
                        "Contrast", min_value=-100, max_value=100, value=0, step=5
                    )
                else:
                    brightness = contrast = 0

                # Gaussian Blur
                if st.sidebar.checkbox("Gaussian Blur"):
                    blur = st.sidebar.slider(
                        "Blur", min_value=1, max_value=31, value=1, step=2
                    )
                else:
                    blur = 0

                if st.checkbox("Resize"):
                    height = st.number_input(
                        "Height", min_value=32, value=1024, step=32
                    )
                    width = st.number_input("Width", min_value=32, value=1024, step=32)
                    transform_list.append(A.Resize(height=height, width=width, p=1))

                # Center Crop
                if st.sidebar.checkbox("Center Crop"):
                    height = st.sidebar.number_input(
                        "Height", min_value=32, value=512, step=32
                    )
                    width = st.sidebar.number_input(
                        "Width", min_value=32, value=512, step=32
                    )
                    transform_list.append(A.CenterCrop(height=height, width=width, p=1))

                transform = A.Compose(
                    transform_list,
                    bbox_params=A.BboxParams(
                        format="pascal_voc", label_fields=["labels"]
                    ),
                )

        if filename:
            while len(filename) < 4:
                filename = "0" + filename

            filtered_df = self.anno_df[
                self.anno_df["image_id"] == f"test/{filename}.jpg"
            ].reset_index(drop=True)
            filtered_df = self.filter_annotations(filtered_df, confidence_threshold)

            image_path = f"{self.folder_path}/test/{filename}.jpg"
            if not os.path.exists(image_path):
                st.error(f"File {filename} does not exist.")
                return

            original_image = self._load_image(image_path)

            if filtered_df.empty:
                transformed_image = original_image
                transformed_df = pd.DataFrame(columns=filtered_df.columns)
            else:
                transformed_image, transformed_df = self.apply_transform(
                    original_image, filtered_df, transform, brightness, contrast, blur
                )

            # activate 열의 데이터 타입을 bool로 변경
            transformed_df["activate"] = transformed_df["activate"].astype(bool)

            if "activation_status" not in st.session_state or len(
                st.session_state.activation_status
            ) != len(transformed_df):
                st.session_state.activation_status = transformed_df["activate"].tolist()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Annotation Information")

                if not transformed_df.empty:
                    display_df = transformed_df.copy()
                    display_df["activate"] = st.session_state.activation_status

                    edited_df = st.data_editor(
                        display_df.drop(columns=["image_id"]),
                        column_config={
                            "activate": st.column_config.CheckboxColumn(
                                "Activate",
                                help="Toggle to activate/deactivate the bounding box",
                                default=True,
                            )
                        },
                        disabled=[
                            "class_name",
                            "class_id",
                            "x_min",
                            "y_min",
                            "x_max",
                            "y_max",
                            "confidence",
                        ],
                        height=400,
                        key="data_editor",
                    )

                    if st.button("Activate All"):
                        new_state = not all(st.session_state.activation_status)
                        st.session_state.activation_status = [new_state] * len(
                            transformed_df
                        )
                        st.rerun()

                    if (
                        st.session_state.activation_status
                        != edited_df["activate"].tolist()
                    ):
                        st.session_state.activation_status = edited_df[
                            "activate"
                        ].tolist()
                        st.rerun()
                else:
                    st.write(
                        "No bounding boxes found for the given confidence threshold."
                    )

                st.markdown("")
                st.markdown("")
                st.markdown("")

                st.markdown(
                    """
                            <style>
                            .custom-button {
                            color: white;
                            padding: 1px 1px;
                            margin: 2px 0px;
                            border: 1px 1px;
                            cursor: none;
                            width: 36.4%;
                            }
                            .red-button {
                                background-color: #ff0000;
                            }
                            .lime-button {
                                background-color: #00ff00;
                                color: #000000;
                            }
                            .blue-button {
                                background-color: #0000ff;
                            }
                            .yellow-button {
                                background-color: #ffff00;
                                color: #000000;
                            }
                            .orange-button {
                                background-color: #ffa500;
                                color: #000000;
                            }
                            .black-button {
                                background-color: #1b1b1b;
                            }
                            .cyan-button {
                                background-color: #00ffff;
                                color: #000000;
                            }
                            .magenta-button {
                                background-color: #ff00ff;
                            }
                            .brown-button {
                                background-color: #a52a2a;
                            }
                            .pink-button {
                                background-color: #ffc0cb;
                                color: #000000;
                            }
                            </style>
                            <button class="custom-button red-button">General Trash</button>
                            <button class="custom-button lime-button">Paper</button>
                            <button class="custom-button blue-button">Paper Pack</button>
                            <button class="custom-button yellow-button">Metal</button>
                            <button class="custom-button orange-button">Glass</button>
                            <button class="custom-button black-button">Plastic</button>
                            <button class="custom-button cyan-button">Styrofoam</button>
                            <button class="custom-button magenta-button">Plastic bag</button>
                            <button class="custom-button brown-button">Battery</button>
                            <button class="custom-button pink-button">Clothing</button>
                            """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.subheader("Prediction")
                if not transformed_df.empty:
                    final_image = self.draw_image(transformed_image, edited_df)
                    st.image(final_image, use_column_width=True)
                else:
                    st.image(original_image)


if __name__ == "__main__":
    folder_path = dataset_path
    csv_path = test_inf_csv_path
    viewer = DataViewer(folder_path, csv_path)
    viewer.run()
