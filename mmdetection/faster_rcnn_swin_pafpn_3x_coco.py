model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
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
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
        )),
    neck=dict(
        type='PAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.7, 1.0, 1.5],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = [
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
    'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type='Flip', p=0.5),
    dict(type='RandomRotate90', p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.1,
        contrast_limit=0.15,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=10,
        p=0.5),
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
        type='Albu',
        transforms=[
            dict(type='Flip', p=0.5),
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
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
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/cv3_train_pesudo.json',
        img_prefix='/opt/ml/detection/dataset/',
        classes=[
            'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
            'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
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
                type='Albu',
                transforms=[
                    dict(type='Flip', p=0.5),
                    dict(type='RandomRotate90', p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.1,
                        contrast_limit=0.15,
                        p=0.5),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=10,
                        p=0.5),
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
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/cv2_val_3.json',
        img_prefix='/opt/ml/detection/dataset/',
        classes=[
            'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
            'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
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
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/test.json',
        img_prefix='/opt/ml/detection/dataset/',
        classes=[
            'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
            'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
        ],
        pipeline=[
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
        ]))
evaluation = dict(
    interval=1, metric='bbox', save_best='bbox_mAP_50', classwise=True)
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=611,
    warmup_ratio=0.001,
    periods=[1, 12, 12, 12],
    restart_weights=[1, 1, 0.5, 0.5],
    min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            interval=500,
            init_kwargs=dict(
                project='objectdetection',
                entity='boostcampaitech3',
                name='faster_rcnn_pseudo3'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
fp16 = dict(loss_scale=512.0)
work_dir = '/opt/ml/detection/baseline/mmdetection/work_dirs/faster_rcnn/faster_rcnn_pseudo3'
auto_resume = False
gpu_ids = [0]
