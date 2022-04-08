# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]


checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval=400),
        dict(type='WandbLoggerHook',interval=200,
            init_kwargs=dict(
                project='objectdetection',
                # entity = 'objectdetection',
                name = 'testing'
            ),
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1)]
workflow = [('train', 1), ('val', 1)]