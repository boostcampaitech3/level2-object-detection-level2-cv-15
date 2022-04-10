# \[AI Tech 3기 Level 2 P Stage\] Object Detection
![image](https://user-images.githubusercontent.com/57162812/162612834-59a7c3ee-4e71-4929-881d-3dbc6bc2e1c0.png)

## Team GAN찮아요 (CV-15) 🎈

김규리_T3016|박정현_T3094|석진혁_T3109|손정균_T3111|이현진_T3174|임종현_T3182|
:-:|:-:|:-:|:-:|:-:|:-:|
|||||

## Final Score 🏅

- Public mAP 0.6975 → Private mAP **0.6827**
- Public 11위/19팀 → Private **11위**/19팀

![image](https://user-images.githubusercontent.com/57162812/162613718-c2a7bd73-774f-4d7f-a8d0-672ec731680c.png)

# Overview

- ***Problem Type.*** 사진 내 쓰레기 10종에 대한 Object Detection
- ***Metric.*** mAP50(Mean Average Precision)
- ***Data.*** COCO format 9754 images (1024, 1024) 10 classes
- ***Class.*** General trash, Paper, Paper pack, Metal, Glass, 
          Plastic, Styrofoam, Plastic bag, Battery, Clothing
- ***Model.*** FasterRCNN, CascadeRCNN, YOLOv5, Universenet
- ***backbone.*** Swin Transformer, PVT2, ResNet
- ***Method.*** Stratified K-fold, Pseudo labeling, Ensemble
- ***Augmentation.*** Mosaic, Multi-scale, TTA

# **Archive contents**

```python
level2-object-detection-level2-cv-15
├──EDA/
├──dataset/
│  ├──data_cleansing/
│  ├──pseudo/
│  └──stratified/
├──models/
│  ├──SoftTeacher/
│  ├──UniversNet/
│  ├──efficientdet-pytorch/
│  ├──mmdetection/              #cascadeRCNN
│  └──yolov5/
└──utils/
   ├──Ensemble.ipynb
   ├──dataVisualization.ipynb
   └──skf.py
```

# Requirements

```jsx
pip install -r requirements.txt
```

# Dataset

### 1. cross validation

이미지와 category가 균등하게 나눠지도록

```
python ./utils/skf.py --path {train dataset path}
```

## 2. 데이터 전수조사

1. 인당 800장씩 4883장 조사
2. 임의의 기준으로 판단하지 않고, 데이터의 유사도를 고려해서 수정
3. object에 bbox가 없을지 annotation 추가 혹은 [https://cleanup.pictures/](https://cleanup.pictures/) 에서 해당 object 삭제
4. 잘못된 annotation 수정 및 삭제

[전수조사 결과](https://docs.google.com/spreadsheets/d/1ZHDPXaJsifjHqrIRRUDGXL_UoGHWZPOl_7RcMT0D6Ik/edit#gid=532025084) 
[수정 예시](https://www.notion.so/Wrap-Up-report-3b4562fffcd744308ef379660a1b0b62)

# Modeling

# TTA & PseudoLabeling

- **TTA**
    - test시에 `flip`과 `resize`와 같은 augmentation을 적용시켜 다음과 같이 앙상블의 효과를 낼 수 있다.
        
        ```python
           Input
             |            # input batch of images
         / / /|\ \ \      # apply augmentations (flips, rotation, scale, etc.)
        | | | | | | |     # pass augmented batches through model
        | | | | | | |     # reverse transformations for each batch of masks/labels
         \ \ \ / / /      # merge predictions (mean, max, gmean, etc.)
             |            # output batch of masks/labels
           Output
        ```
        
- **PseudoLabeling**
    - 모델이 예측한 label들 중 특정 confidence score값 이상만을 Pseudo label이라 간주하고 train데이터와 합치는 방식
    - 사용법
        
        `python ./utils/make_pseudo.py --train {train의 json파일} --pseudo {모델이 예측한 csv파일} --output {결과물 json파일 경로} --threshold {confidence score제한값}`
        
    - 예시
        
        `python ./utils/make_pseudo.py --train {train.json} --pseudo {for_pseudo.csv} --output {output_test.json} --threshold {0.3}`
        

# Ensemble

- `ensemble_boxes` 라이브러리 사용
    
    ```python
    pip install ensemble_boxes
    ```
    
    - nms
    - soft_nms
    - non_maximum_weighted
    - weighted_boxes_fusion

- 주로 `weighted_boxes_fusion` 방식을 사용하였다.
    - ***iou_thr*** = 0.4
    - ***skip_box_thr***=0
    - ***conf_type*** = 'avg'

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20022ac6-9f4a-49ec-be1e-f6263eccf758/Untitled.png)

- 다양한 모델 및 다양한 dataset 사용
    - ***dataset*** : 5fold, raw data vs. cleansing data
    - ***mode***l : `cascade_rcnn`, `universenet`, `yolov5`
