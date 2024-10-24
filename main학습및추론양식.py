#inference 양식

python main.py \
    --train 0 \
    --config_dir /data/ephemeral/home/level2-objectdetection-cv-12/baseline/mmdetection/custom_configs/DDQ/DDQ_VAL.py \
    --load_from /data/ephemeral/home/level2-objectdetection-cv-12/baseline/mmdetection/work_dirs/DDQ_newfold/epoch_9.pth \

# trian 양식

python main.py \
    --train 1 \
    --config_dir /data/ephemeral/home/level2-objectdetection-cv-12/baseline/mmdetection/custom_configs/DDQ/DDQ.py \
    --wandb_name DDQ_JIHWAN \
    --amp

# 이어서 학습하기 양식

python main.py \
    --config_dir /data/ephemeral/home/Lv2.Object_Detection/baseline/mmdetection/custom_configs/DINO/dino-5scale_swin-l_8xb2-12e_trash_yjh.py \
    --wandb_name DINO_JIHWAN \
    --amp \
    --resume_from /path/to/your/checkpoint.pth
