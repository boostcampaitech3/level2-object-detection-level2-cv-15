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

# # disable opencv multithreading to avoid system being overloaded
# opencv_num_threads = 0
# # set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'



#https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-4
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', interval=100),
        dict(type='WandbLoggerHook',interval=100,
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
# 1 epoch에 train과 validation을 모두 하고 싶으면 workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# #정현님 버전
# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='WandbLoggerHook',interval=1000,
#             init_kwargs=dict(
#                 project='test-project',
#                 entity = 'winner',
#                 name = 'MMDetection_cascade_rcnn_swin_s'
#             ),
#             )
#     ])
# # yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]

# # disable opencv multithreading to avoid system being overloaded
# opencv_num_threads = 0
# # set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'