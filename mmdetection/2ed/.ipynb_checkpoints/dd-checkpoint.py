# dataset settings
# from mmdetection/configs/_base_/datasets/coco_detection.py
dataset_type = "CocoDataset"
data_root = "/opt/ml/detection/dataset/"
classes = [
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

img_norm_cfg = dict(
    mean=[127.49413776397705, 127.43779182434082, 127.46098327636719],
    std=[73.86627551077616, 73.88234865304638, 73.8944344154546],
    to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    # dict(
    #     type="AutoAugment",
    #     policies=[
    #         [
    #             dict(
    #                 type="Resize",
    #                 img_scale=[
    #                     (512, 768),
    #                     (640, 768),
    #                     (768, 768),
    #                     (768, 512),
    #                     (768, 640),                       
    #                 ],
    #                 multiscale_mode="value",
    #                 keep_ratio=True,
    #             )
    #         ],
    #         [
    #             dict(
    #                 type="Resize",
    #                 img_scale=[(512, 1024), (640, 1024), (768, 1024)],
    #                 multiscale_mode="value",
    #                 keep_ratio=True,
    #             ),
    #             dict(
    #                 type="RandomCrop",
    #                 crop_type="absolute_range",
    #                 crop_size=(256, 256),
    #                 allow_negative_crop=True,
    #             ),
    #             dict(
    #                 type="Resize",
    #                 img_scale=[
    #                     (512, 768),
    #                     (640, 768),
    #                     (768, 768),
    #                     (768, 512),
    #                     (768, 640), 
    #                 ],
    #                 multiscale_mode="value",
    #                 override=True,
    #                 keep_ratio=True,
    #             ),
    #         ],
    #     ],
    # ),
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
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=7,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "cv_train_1.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "cv_val_1.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
    ),
)
evaluation = dict(interval=1, metric="bbox", classwise = True, save_best="bbox_mAP_50")