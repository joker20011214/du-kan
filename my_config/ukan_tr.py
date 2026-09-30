norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoderFull',
    pretrained=None,
    decode_head=dict(
        type='UKAN',
        tr=True,
        channels=32,
        num_classes=8,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=2.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(128, 128))
)

dataset_type = 'MyDataset'
img_size = (512, 512)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = "./datasets/"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/balanced_images',
            ann_dir='train/balanced_labels',
            pipeline=train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
# ---- 学习率策略：增加warmup ----
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
    warmup='linear', warmup_iters=1500, warmup_ratio=1e-3)
runner = dict(
    type='IterBasedRunner',
    max_iters=20000)
checkpoint_config = dict(
    by_epoch=False,
    save_optimizer=False,
    interval=20000)
evaluation = dict(
    interval=1000,
    metric=['mIoU', 'mFscore', 'mDice'])
work_dir = f'./work_dirs/{{ fileBasenameNoExtension }}'