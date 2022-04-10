_base_ = [
    'models/universenet101_2008d.py',
    '_base_/datasets/coco_detection_16.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]
data_root = '../../dataset/'
data = dict(samples_per_gpu=4,
            train=dict(ann_file=data_root +  'cv3_train_pesudo.json'),
            val=dict(ann_file=data_root +  'cv2_val_3.json'))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
fp16 = dict(loss_scale=512.)

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=500,
            init_kwargs=dict(
                project='objectdetection',
                entity = 'boostcampaitech3',
                name = 'Univ2008d_20_3'
            ))
    ])

tta_scale = [(1024, 1024), (2048, 2048)]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=tta_scale,
        flip=True,
        # flip_direction = ['horizontal','vertical', "diagonal"],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(test=dict(pipeline=test_pipeline))
