_base_ = [
    'models/universenet101_2008d.py',
    '_base_/datasets/coco_detection_16.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)

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
                name = 'Univ2008d_20'
            ))
    ])

tta_scale = [(512,512), (600,600), (700, 700), (800, 800), (900, 900),
             (1024, 1024), (1200, 1200), (1300, 1300), (1400, 1400),
             (1500, 1500), (1800, 1800), (2048, 2048), (2200, 2200)]
# scale_ranges = [(96, 10000), (96, 10000), (64, 10000), (64, 10000),
#                 (64, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 256),
#                 (0, 256), (0, 192), (0, 192), (0, 96)]

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
