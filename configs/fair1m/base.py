# model settings
_base_ = "../base.py"
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[

            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000, 160000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=10)
evaluation = dict(interval=4000)

fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ],
)
