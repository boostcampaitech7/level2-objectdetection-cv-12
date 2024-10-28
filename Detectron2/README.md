<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

<a href="https://opensource.facebook.com/support-ukraine">
  <img src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" alt="Support Ukraine - Help Provide Humanitarian Aid to Ukraine." />
</a>

<br>

# Results on Trash-Dataset in Detectron2

|Model|AP|AP50|AP75|APs|APm|APl|mAP(Public)|mAP(Private)
|---|---|---|---|---|---|---|---|---|
Cascade_mask_rcnn_mvitv2|47.983|63.247|50.892|0.579|12.670|55.868|0.6513|0.6372
EVA|59.060|70.148|61.067|4.917|14.866|67.322|0.6827|0.6700

<br>

# How to Run

#### Cascade_mask_rcnn_mvitv2
> Cascade_mask_rcnn_mvitv2 실행  : python /detectron2/tools/lazyconfig_train_net.py --config-file detectron2/projects/MViTv2/configs cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x.py

#### EVA
> EVA-02/det 실행 : python /path/to/your/lazyconfig_train_net.py --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py


<br>
### Experiments
