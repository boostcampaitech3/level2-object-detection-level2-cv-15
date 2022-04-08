# https://github.com/boostcampaitech2/object-detection-level2-cv-04/blob/main/mmdetection/configs/finals/77_cascade_swins_fpn_3x_TTA.py#L15

##############################
# Model:    Swin-S           #
# Backbone: Swin-T           #
# Neck:     FPN              #
# Head:     Cascade R-CNN    #
# Opt:      AdamW            #
# LR:       0.0001           #
# Sch:      CosineRestart    #
# Epoch:    36               #
# Batch:    8                #
##############################

# merge configs
# _base_ = [
#     './configs/_base_/models/cascade_rcnn_r50_fpn.py',
#     './2ed/dataset_cascade_swins.py',
#     './configs/_base_/default_runtime.py',
#     './2ed/schedule_3x_cosinerestart.py'
# ]

_base_ = [
    './configs/_base_/models/cascade_rcnn_r50_fpn.py',
    # './coco_detection.py',
    './2ed/dd.py',
    './configs/_base_/default_runtime.py',
    './configs/_base_/schedules/schedule_1x.py'
]


# _base_ = [
#     './configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
#     # './coco_detection.py',
#     # './configs/_base_/default_runtime.py',
#     # './configs/_base_/schedules/schedule_1x.py'
# ]




# Load pretrained Swin-S model
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

# set model backbone to Swin-S
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# # Mixed Precision training
# fp16 = dict(loss_scale=512.)


# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# # augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='AutoAugment',
#         policies=[[
#             dict(
#                 type='Resize',
#                 img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                            (736, 1333), (768, 1333), (800, 1333)],
#                 multiscale_mode='value',
#                 keep_ratio=True)
#         ],
#                   [
#                       dict(
#                           type='Resize',
#                           img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                           multiscale_mode='value',
#                           keep_ratio=True),
#                       dict(
#                           type='RandomCrop',
#                           crop_type='absolute_range',
#                           crop_size=(384, 600),
#                           allow_negative_crop=True),
#                       dict(
#                           type='Resize',
#                           img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                      (576, 1333), (608, 1333), (640, 1333),
#                                      (672, 1333), (704, 1333), (736, 1333),
#                                      (768, 1333), (800, 1333)],
#                           multiscale_mode='value',
#                           override=True,
#                           keep_ratio=True)
#                   ]]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# data = dict(train=dict(pipeline=train_pipeline))



##swin/Mask_rcnn_swin-t-p4-w7_fpn_ms-crop ~~ 에서 가져옴
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
# lr_config = dict(warmup_iters=1000, step=[8, 12])
# runner = dict(max_epochs=36)



# base schedule 에서 가져옴
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=10)