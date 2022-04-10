# \[AI Tech 3기 Level 2 P Stage\] Object Detection
![image](https://user-images.githubusercontent.com/57162812/162612834-59a7c3ee-4e71-4929-881d-3dbc6bc2e1c0.png)

## Team GAN찮아요 (CV-15) 🎈

김규리|박정현|석진혁|손정균|이현진|임종현|
:-:|:-:|:-:|:-:|:-:|:-:|
|||||

## Final Score 🏅

- Public mAP 0.7006 → Private mAP **0.6836**
- Public 11위 → Private **11위**

![image](https://user-images.githubusercontent.com/57162812/162613718-c2a7bd73-774f-4d7f-a8d0-672ec731680c.png)


# readme

sdflsdkflsdkfj

# 안녕하세요?

## 안녕하세요??

### 안녕하세요??

1. 안녕?

- 안녕?
# TTA & PseudoLabeling

- TTA
    - flip, resize
PseudoLabeling
- PseudoLabeling
    - 모델이 예측한 label들 중 특정 confidence score값 이상만을 Pseudo label이라 간주하고 train데이터와 합치는 방식
    - 사용법
        
        `python make_pseudo.py --train {train의 json파일} --pseudo {모델이 예측한 csv파일} --output {결과물 json파일 경로} --threshold {confidence score제한값}`
        
    - 예시
        
        `python make_pseudo.py --train {train.json} --pseudo {for_pseudo.csv} --output {output_test.json} --threshold {0.3}`
        PseudoLabeling
