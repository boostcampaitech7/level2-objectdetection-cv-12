import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
import matplotlib.font_manager as fm
from PIL import ImageFont

################################################
# train dataset 시각화를 streamlit을 통해 진행  ##
# 터미널에서 `streamlit HOME.py`로 실행  ##
# path만 변경하여 사용 ##
################################################

dataset_path = "Enter your dataset path"
test_inf_csv_path ="Enter your test data inference csv file path"
val_inf_csv_path = "Enter your validation data inference csv file path"

val_inf_csv_path1 = "Enter your first validation data inference csv file path for ensemble"
val_inf_csv_path2 = "Enter your second validation data inference csv file path for ensemble"


if __name__ == "__main__":
    st.set_page_config(layout="centered")

    with st.sidebar:
        st.header("Menu")

    markdown_text_1 = """
    ### 소개
    바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.
    """

    st.markdown(markdown_text_1)

    image_path = "Intro.png"
    st.image(image_path, use_column_width=True)

    markdown_text_2 = """
    분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

    따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

    여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

    **Input** : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 됩니다. bbox annotation은 COCO format으로 제공됩니다. (COCO format에 대한 설명은 학습 데이터 개요를 참고해주세요.)

    **Output** : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요.)
    """

    st.markdown(markdown_text_2)
