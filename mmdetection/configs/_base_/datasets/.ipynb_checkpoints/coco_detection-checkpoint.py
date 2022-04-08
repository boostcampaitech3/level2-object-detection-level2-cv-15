# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
data_root = "/opt/ml/detection/dataset/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.json',
#         img_prefix=data_root + 'train2017/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')



data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cv_train_1.json',
        # ann_file=data_root + 'train.json',
        img_prefix=data_root ,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

#     classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
#                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
#     root='../../dataset/' 
    
    

#     # dataset config 수정
#     # cfg.data.train.classes = classes
#     # cfg.data.train.img_prefix = root
#     # cfg.data.train.ann_file = root + 'train.json' # train json 정보
#     # cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

#     ##soft teacher
#     cfg.data.train.sup.img_prefix = root
#     cfg.data.train.sup.ann_file = root + 'train.json' # train json 정보
#     cfg.data.train.unsup.img_prefix = root
#     cfg.data.train.unsup.ann_file = root + 'test.json' # train json 정보



#     cfg.data.test.classes = classes
#     cfg.data.test.img_prefix = root
#     cfg.data.test.ann_file = root + 'test.json' # test json 정보
#     # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
