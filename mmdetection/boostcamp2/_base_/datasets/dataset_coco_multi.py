dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset2/'
# class settings
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Albumentations transforms settings
albu_train_transforms = [
    dict(type='Flip',p=0.5),
    dict(type='RandomRotate90',p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ],
    p=0.1)
]


# set train_pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),

    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (614, 1024),
                        (655, 1024),
                        (696, 1024),
                        (737, 1024),
                        (778, 1024),
                        (819, 1024),
                        (860, 1024),
                        (901, 1024),
                        (942, 1024),
                        (983, 1024),
                        (1024, 1024),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(512, 1024), (640, 1024), (768, 1024)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(256, 256),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (614, 1024),
                        (655, 1024),
                        (696, 1024),
                        (737, 1024),
                        (778, 1024),
                        (819, 1024),
                        (860, 1024),
                        (901, 1024),
                        (942, 1024),
                        (983, 1024),
                        (1024, 1024),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),

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
            'gt_bboxes': 'bboxes',
        },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# set valid pipeline
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# set test_pipeline for TTA(Test Time Augmentation)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        flip=True,
        img_scale=[(512, 1024), (1024, 1024)],
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

# data settings
data = dict(
    samples_per_gpu=8, # batch size for GPU
    workers_per_gpu=2, # num_workers
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cv2_train_1.json',  # use entire train dataset
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cv2_val_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

# evaluation settings
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50", classwise = True)