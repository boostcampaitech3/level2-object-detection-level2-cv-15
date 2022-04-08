dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = [
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
    'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
img_norm_cfg = dict(
    mean=[127.49413776397705, 127.43779182434082, 127.46098327636719],
    std=[73.86627551077616, 73.88234865304638, 73.8944344154546],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(614, 1024), (655, 1024), (696, 1024), (737, 1024),
                          (778, 1024), (819, 1024), (860, 1024), (901, 1024),
                          (942, 1024), (983, 1024), (1024, 1024)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(512, 1024), (640, 1024), (768, 1024)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (256, 256),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(614, 1024), (655, 1024), (696, 1024),
                                    (737, 1024), (778, 1024), (819, 1024),
                                    (860, 1024), (901, 1024), (942, 1024),
                                    (983, 1024), (1024, 1024)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='Normalize',
        mean=[127.49413776397705, 127.43779182434082, 127.46098327636719],
        std=[73.86627551077616, 73.88234865304638, 73.8944344154546],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[
                    127.49413776397705, 127.43779182434082, 127.46098327636719
                ],
                std=[73.86627551077616, 73.88234865304638, 73.8944344154546],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=7,
    workers_per_gpu=3,
    train=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/cv_train_1.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale': [(614, 1024), (655, 1024), (696, 1024),
                                  (737, 1024), (778, 1024), (819, 1024),
                                  (860, 1024), (901, 1024), (942, 1024),
                                  (983, 1024), (1024, 1024)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type': 'Resize',
                              'img_scale': [(512, 1024), (640, 1024),
                                            (768, 1024)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (256, 256),
                              'allow_negative_crop': True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(614, 1024), (655, 1024),
                                            (696, 1024), (737, 1024),
                                            (778, 1024), (819, 1024),
                                            (860, 1024), (901, 1024),
                                            (942, 1024), (983, 1024),
                                            (1024, 1024)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(
                type='Normalize',
                mean=[
                    127.49413776397705, 127.43779182434082, 127.46098327636719
                ],
                std=[73.86627551077616, 73.88234865304638, 73.8944344154546],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    val=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/cv_val_1.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            127.49413776397705, 127.43779182434082,
                            127.46098327636719
                        ],
                        std=[
                            73.86627551077616, 73.88234865304638,
                            73.8944344154546
                        ],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    test=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/test.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            127.49413776397705, 127.43779182434082,
                            127.46098327636719
                        ],
                        std=[
                            73.86627551077616, 73.88234865304638,
                            73.8944344154546
                        ],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')))
evaluation = dict(
    interval=1, metric='bbox', classwise=True, save_best='bbox_mAP_50')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', interval=100),
        dict(
            type='WandbLoggerHook',
            interval=100,
            init_kwargs=dict(
                project='objectdetection', name='dynamic_head_ATSS_f5'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='ATSS',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )),
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
work_dir = './work_dirs/dynamic/5'
auto_resume = False
gpu_ids = [0]
