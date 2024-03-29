# model settings
_base_ = "base.py"

model = dict(
    type='FasterRCNNOBB',

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(
            type="Pretrained", checkpoint="modelzoo://resnet50"
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
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
            type='SharedFCBBoxHeadRbbox',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            target_means=[0., 0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
            reg_class_agnostic=False,
            with_module=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssignerCy',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerCy',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
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
            nms_across_levels=False,
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            # score_thr=0.05, nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img=1000)
            score_thr=0.05, nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img=2000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    )
)
# model training and testing settings
semi_wrapper = dict(

)
# dataset settings
tile_side_len = 1000
dataset_type = 'DOTA1_5Dataset_v2'
data_root = f'gaofen_data/fair1_{tile_side_len}/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=3,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type=dataset_type,
            ann_file=data_root + f'train{tile_side_len}/few_shot_8.json',
            img_prefix=data_root + f'train{tile_side_len}/images/',
        ),
        unsup=dict(
            type=dataset_type,
            ann_file=data_root + f'train{tile_side_len}/train1000_5classes.json',
            img_prefix=data_root + f'train{tile_side_len}/images/',

        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + f'val{tile_side_len}/few_shot_50.json',
        img_prefix=data_root + f'val{tile_side_len}/images',

    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + f'val{tile_side_len}/few_shot_50.json',
        img_prefix=data_root + f'val{tile_side_len}/images',
    ),

    sampler=dict(
        train=dict(
            sample_ratio=[1, 2],
        )
    ),
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[8, 11],

)
checkpoint_config = dict(interval=1000)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ]
)
# yapf:enable
# runtime settings
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=100000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_obb_r50_fpn_1x_fair1m_semi_supervised_v3'
load_from = None
resume_from = None
workflow = [('train', 1)]
