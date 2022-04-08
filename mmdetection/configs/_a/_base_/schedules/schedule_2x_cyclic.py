# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 1e-4),
    cyclic_times=24,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.5 / 0.9, 1),
    cyclic_times=0.5,
    step_ratio_up=0.4,
)
runner = dict(type='EpochBasedRunner', max_epochs=24)
