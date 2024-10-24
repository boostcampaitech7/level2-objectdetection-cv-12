import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
import matplotlib.font_manager as fm
from PIL import ImageFont
import albumentations as A
import cv2
import os, sys
from HOME import dataset_path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
st.set_page_config(layout="wide")


class DataViewer:
    def __init__(self, folder_path):
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
        self.anno_df = self.json2df(self.folder_path, self.classes)

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

        train_df = pd.DataFrame(
            {
                "image_id": image_ids,
                "class_name": class_name,
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "activate": True,
            }
        )

        return train_df

    @staticmethod
    @st.cache_data
    def get_font(size=25):
        font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        return ImageFont.truetype(font_path, size)

    def draw_image(self, image, annotations):
        """Draw bounding boxes and labels on image"""
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
                        f"{idx} {row['class_name']}",
                        fill=color,
                        font=font,
                    )

        return image

    @staticmethod
    @st.cache_data
    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        brightness = int(brightness * 255 / 100)
        contrast = float(contrast * 0.01)

        image = cv2.convertScaleAbs(image, alpha=contrast + 1, beta=brightness)
        return image

    @staticmethod
    @st.cache_data
    def apply_gaussian_blur(image, blur):
        return cv2.GaussianBlur(image, (blur, blur), 0)

    def apply_transform(
        self, image, annotations, transform, brightness, contrast, blur
    ):
        image = np.array(image)
        image = self.adjust_brightness_contrast(image, brightness, contrast)

        if blur > 0:
            image = self.apply_gaussian_blur(image, blur)

        # Prepare bboxes and labels for albumentations
        bboxes = annotations[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        labels = annotations["class_id"].tolist()

        # Apply the transformation
        transformed = transform(image=image, bboxes=bboxes, labels=labels)

        # Update the annotations DataFrame
        new_bboxes = pd.DataFrame(
            transformed["bboxes"], columns=["x_min", "y_min", "x_max", "y_max"]
        )
        new_labels = pd.Series(transformed["labels"], name="class_id")

        # Keep only valid bounding boxes (those with positive width and height)
        valid_boxes = (new_bboxes["x_max"] > new_bboxes["x_min"]) & (
            new_bboxes["y_max"] > new_bboxes["y_min"]
        )
        new_bboxes = new_bboxes[valid_boxes]
        new_labels = new_labels[valid_boxes]

        # Update annotations with valid bounding boxes
        annotations = annotations.iloc[: len(new_bboxes)].copy()
        annotations[["x_min", "y_min", "x_max", "y_max"]] = new_bboxes
        annotations["class_id"] = new_labels
        annotations["class_name"] = annotations["class_id"].map(
            lambda x: self.classes[int(x)]
        )

        return Image.fromarray(transformed["image"]), annotations

    @staticmethod
    @st.cache_data
    def load_image(image_path):
        return Image.open(image_path)

    def run(self):
        with st.sidebar:
            st.header("Menu")
            filename = st.text_input("File name")

            if filename:
                st.write("Transform")

                transform_list = []

                # Existing transformations
                if st.sidebar.checkbox("HorizontalFlip"):
                    transform_list.append(A.HorizontalFlip(p=1))
                if st.sidebar.checkbox("VerticalFlip"):
                    transform_list.append(A.VerticalFlip(p=1))

                # Rotation
                if st.sidebar.checkbox("Rotate"):
                    angle = st.sidebar.slider(
                        "Rotation Angle", min_value=-30, max_value=30, value=0, step=1
                    )
                    transform_list.append(
                        A.Rotate(limit=(angle, angle), border_mode=2, value=0, p=1)
                    )

                # Brightness and Contrast
                if st.sidebar.checkbox("Brightness & Contrast"):
                    brightness = st.sidebar.slider(
                        "Brightness", min_value=-100, max_value=100, value=0, step=5
                    )
                    contrast = st.sidebar.slider(
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

                # Resize
                if st.sidebar.checkbox("Resize"):
                    height = st.sidebar.number_input(
                        "Height", min_value=32, value=1024, step=32
                    )
                    width = st.sidebar.number_input(
                        "Width", min_value=32, value=1024, step=32
                    )
                    transform_list.append(A.Resize(height=height, width=width, p=1))

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
                self.anno_df["image_id"] == f"train/{filename}.jpg"
            ].reset_index(drop=True)

            image_path = f"{self.folder_path}/train/{filename}.jpg"
            if not os.path.exists(image_path):
                st.error(f"File {filename} does not exist.")
                return

            original_image = self.load_image(image_path)

            # Apply transformations
            transformed_image, transformed_df = self.apply_transform(
                original_image, filtered_df, transform, brightness, contrast, blur
            )

            # Initialize or update session state for activation status
            if "activation_status" not in st.session_state or len(
                st.session_state.activation_status
            ) != len(transformed_df):
                st.session_state.activation_status = transformed_df["activate"].tolist()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Annotation Information")

                # Create a copy of the dataframe with the current activation status
                display_df = transformed_df.copy()
                display_df["activate"] = st.session_state.activation_status

                # Use st.data_editor for editable dataframe
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
                    ],
                    height=400,
                    key="data_editor",
                )

                # 'Activate All' button logic
                if st.button("Activate All"):
                    new_state = not all(st.session_state.activation_status)
                    st.session_state.activation_status = [new_state] * len(
                        transformed_df
                    )
                    st.rerun()

                # Update session state if changes are detected
                if st.session_state.activation_status != edited_df["activate"].tolist():
                    st.session_state.activation_status = edited_df["activate"].tolist()
                    st.rerun()

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
                st.subheader("Ground Truth")
                final_image = self.draw_image(transformed_image, edited_df)
                st.image(final_image, use_column_width=True)


if __name__ == "__main__":
    folder_path = dataset_path
    viewer = DataViewer(folder_path)
    viewer.run()
