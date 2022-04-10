_base_ = [
    'models/universenet101_2008d.py',
    '_base_/datasets/coco_detection_mstrain_480_960.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)

optimizer = dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

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
                project='test-project',
                entity = 'winner',
                name = 'Univ2008d_Adam_cos'
            ))
    ])

runner = dict(max_epochs=40)