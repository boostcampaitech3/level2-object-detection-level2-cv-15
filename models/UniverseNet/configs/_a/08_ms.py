_base_ = [
    'models/universenet101_2008d.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=500,
            init_kwargs=dict(
                project='objectdetection',
                entity = 'boostcampaitech3',
                name = 'Univ2008d_ms'
            ))
    ])

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


# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

albu_train_transforms = [
    dict(type='RandomRotate90',p=1.0)

]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(512,512), (512,512)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root +  'cv_train_1.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root +  'cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', classwise=True, save_best="bbox_mAP_50")