_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='RPNDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNCenterHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            # ratios=[0.5, 1.0, 2.0],
            # Single anchor <
            ratios=[1.0],
            # >
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            # Delta -vs- LRTB <
            type='TBLRBBoxCoder',
            normalizer=1.0,
            # type='DeltaXYWHBBoxCoder',
            # target_means=[.0, .0, .0, .0],
            # target_stds=[1.0, 1.0, 1.0, 1.0]
            # >
            ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # <
        reg_decoded_bbox=True,  # for IoU Loss
        loss_bbox=dict(type='IoULoss', linear=True, loss_weight=10.0),
        # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        # >
        loss_center=dict(type='L1Loss', loss_weight=1.0),
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
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
            # <
            # Centerness assigner and sampler 
            center_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            center_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=255/256.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            # >
            allowed_border=0,
            pos_weight=-1,
            debug=False)
        ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            # nms_post=1000,
            # max_num=1000,
            # nms_thr=0.7,
            nms_post=2000,
            max_num=200,
            nms_thr=0.5,
            min_bbox_size=0),
        ))

# Dataset
dataset_type = 'CocoSplitDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # <
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        # >
        type=dataset_type,
        pipeline=train_pipeline,
        ),
    val=dict(
        # <
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        # >
        type=dataset_type,),
    test=dict(
        # <
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        # >
        type=dataset_type,))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 7])
total_epochs = 8
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir='work_dirs/oln_lrtb01_center_cls_rpn'