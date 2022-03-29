# train
cd tools

python train.py name <w&b에 찍힐 실험 이름> config <사용할 config파일 경로> train_data <train 데이터 경로> val_data <val 데이터 경로> --work-dir <각 epoch마다 찍힐 로그와 모델이 저장될 폴더 지정>

-> ex) python train.py name "cascade_1" config ../configs/custom/cascade_swin_fpn_3x_ChangedIoU.py train_data /dataset/CV/train_1.json val_data /dataset/CV/val_1.json --work-dir /output/cascade
