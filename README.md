<div align=center>
  <h1> 🚀Yolov5🚀 </h1>
</div>

COCO dataset과 Yolo dataset의 형식은 다르기 때문에 데이터 형식 변환이 필요하다. 이때, `convert2yolo`를 사용한다.

<div align=center>
  <h2>Convert2Yolo</h2>
</div>

```
git clone https://github.com/ssaru/convert2Yolo.git
```

1. root folder인 convert2Yolo에 train folder 생성
2. class 종류를 정의하는 names.txt를 복사
3. command 실행하여 COCO data format json을 yolov5 label로 변환한 파일과 image path를 담고 있는 manifest 파일을 생성 각각 train/val 파일 생성
    ```python
    python3 example.py --datasets {변환 전 dataformat: COCO} --img_path {변환할 이미지 경로} --label {json file 경로} --convert_output_path ./ --img_type ".jpg" --manifest_path ./ --cls_list_file {names.txt 경로}
    ```
그렇다면, 다음과 같이 train folder에 label과 bounding box를 정의하는 txt 파일이 생성된다.

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/159856733-6579e32e-251d-43cb-93d1-37e8836e3e43.png" width = "30%"></p>

> **0005.txt**
> 
> <p align="center"><img src="https://user-images.githubusercontent.com/57162812/159856887-9bf95617-514f-443c-a17a-53fc66967deb.png" width="20%"></p>
> 왼쪽부터 class, x_center, y_center, width, height

<div align=center>
  <h2>Yolov5</h2>
</div>

<p align="center"><img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg" width="50%"></p>

```python 
git clone https://github.com/open-mmlab/mmdetection.git
```

1. convert2yolo로 생성된 txt file을 image가 있는 folder로 변경해준다.
2. train/val 이미지의 경로들을 작성한 txt file을 만들어 준다.
    > **train_1.txt**
    > <p align="center"><img src="https://user-images.githubusercontent.com/57162812/162350254-d718a41a-7113-48a2-bdc9-28d9ac3088a3.png" width="50%"></p>
3. yolov5/data에 custom_data.yaml을 생성해준다. (data 경로 설정 및 class_num 설정)
    ```yaml
    # cutom_data.yaml
    train: {train_1.txt 파일 경로}
    val: {val_1.txt 파일 경로}
    nc: 10
    names: ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', Clothing]
    ```
    
4. yolov5 folder로 directory를 설정 후, train.py를 실행시킨다.
      ```python
       python train.py --data ./data/custom_data.yaml --cfg ./models/yolov5x.yaml --weight yolov5x.pt --batch 16 --workers 4 --epochs 100 --name yolov5x_100
       ```
       - data : custom_data 경로
       - cfg : 사용할 모델 : yolov5/models에서 확인 가능
       - weight : 사용할 모델의 pretained weight
       - best.pt : best mAP 기준, latest.pt : 마지막 model 
5. 생성된 pt file을 사용해 inference를 진행한다.
    '''python
    python detect.py --source /opt/ml/detection/dataset/test --save-txt --save-conf --con-thres {값} --weights {model 저장 경로} --augment
    ```
    - source : test image가 들어있는 folder의 경로
    - save-txt : bbox 좌표 저장
    - save-conf : confidence score 저장
    - con-thres : confidence threshold 지정
    - iou-thres : iou threshold 지정
    - weights : inference 실행할 model의 weight 경로 : pt 파일 : 여러개 작성시 ensemble 
    - augment : tta 
