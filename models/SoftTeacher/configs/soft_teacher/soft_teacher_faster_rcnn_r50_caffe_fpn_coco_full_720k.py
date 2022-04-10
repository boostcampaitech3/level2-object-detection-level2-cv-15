_base_="base.py"
data_root = "/opt/ml/detection/dataset/"
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(

        sup=dict(
            
            # ann_file="data/coco/annotations/instances_train2017.json",
            # img_prefix="data/coco/train2017/",
            classes=classes,
            ann_file=data_root + 'cv_train_1.json',
            # ann_file=data_root + 'train.json',
            img_prefix=data_root ,  

        ),
        unsup=dict(

            # ann_file="data/coco/annotations/instances_unlabeled2017.json",
            # img_prefix="data/coco/unlabeled2017/",
            classes=classes,
            ann_file=data_root + 'test.json',
            img_prefix=data_root,

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=8,
#     train=dict(

#         sup=dict(

#             ann_file="data/coco/annotations/instances_train2017.json",
#             img_prefix="data/coco/train2017/",

#         ),
#         unsup=dict(

#             ann_file="data/coco/annotations/instances_unlabeled2017.json",
#             img_prefix="data/coco/unlabeled2017/",

#         ),
#     ),
#     sampler=dict(
#         train=dict(
#             sample_ratio=[1, 1],
#         )
#     ),
# )

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

# workflow = [('train', 1), ('val', 1)]