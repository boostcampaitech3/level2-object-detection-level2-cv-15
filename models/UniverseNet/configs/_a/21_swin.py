_base_ = [
    'models/universenet_swin.py',
    '_base_/datasets/coco_detection_16.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]

data_root = '../../dataset/'
data = dict(samples_per_gpu=2,
            train=dict(ann_file=data_root +  'cv3_train_pesudo.json'),
            val=dict(ann_file=data_root +  'cv2_val_3.json'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
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
                project='objectdetection',
                entity = 'boostcampaitech3',
                name = 'Univ2008d_21_3_swin'
            ))
    ])

