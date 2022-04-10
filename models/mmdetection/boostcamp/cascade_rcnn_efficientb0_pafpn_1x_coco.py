_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_ = True,
        type='EfficientNet',
        model_type='efficientnet-b0',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
        out_indices=(0, 1, 3, 6)),
    neck = dict(
        type = 'PAFPN',
        in_channels = [24, 40, 112, 1280]
    )
        )

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