_base_ = [
    'models/universenet101_2008d.py',
    '_base_/datasets/coco_detection_albu08.py',
    '_base_/schedules/schedule_2x_cyclic.py', '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     _delete_=True,
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=24,
#     step_ratio_up=0.4,
# )
# momentum_config = dict(
#     _delete_=True,
#     policy='cyclic',
#     target_ratio=(0.85 / 0.95, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )

fp16 = dict(loss_scale=512.)

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=500,
            init_kwargs=dict(
                # project='asd',
                project='objectdetection',
                entity = 'boostcampaitech3',
                name = 'Univ2008d_cyclic'
            ))
    ])
