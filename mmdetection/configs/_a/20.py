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
