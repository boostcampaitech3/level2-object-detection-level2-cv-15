## 1. mmdetection 설치

아래 링크의 튜토리얼에 맞춰서  라이브러리 설치 

https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md

## 2. How to train

### Train

```
python ./tools/train.py --name {project name} --config {train config file path} --work-dir {the dir to save logs and models}
```

### Inference

```
python ./tools/test.py --config {test config file path} --checkpoing {checkpoint file} --work-dir {the dir to save the file}
```

### Ensemble

```
python ./tools/ensemble/ensemble.py --cfg {ensemble config file path}
```



## 3. 실험 결과

### 1) cascade rcnn

### 2) neck, backbone 변화

### 3) results

