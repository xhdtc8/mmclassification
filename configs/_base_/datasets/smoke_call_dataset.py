# dataset settings
dataset_type = 'TwowayDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='RandomCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    # samples_per_gpu=1, # for test
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.4,
        dataset=dict(
                type=dataset_type,
                data_prefix='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/train/',
                ann_file='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/train.txt',
                pipeline=train_pipeline
                )
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/test/',
        ann_file='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/test/',
        ann_file='/data/xuhua/data/kitchen/smoking-calling/smoke-hard-all/test.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metric=['precision', 'recall', 'accuracy'],metric_options = {'topk': (1,)})
