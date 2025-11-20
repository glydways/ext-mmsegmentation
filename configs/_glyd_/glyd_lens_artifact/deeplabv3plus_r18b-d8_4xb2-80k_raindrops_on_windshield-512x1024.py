_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=1,
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        loss_decode=dict(use_sigmoid=True)
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=1, loss_decode=dict(use_sigmoid=True)))

# Train schedule
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=1000)

# Runtime overrides:
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend')]
visualizer = dict(vis_backends=vis_backends)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5,
        save_last=True,
    ),
    visualization=dict(type='CustomSegVisualizationHook', draw=True, interval=2, val_interval=1000, draw_filenames=['D3_0042_img.png', 'D3_1100_img.png', 'D3_1980_img.png', 'D3_0423_img.png', 'Extra1_0000_img.png', 'Extra2_0000_img.png'])
)

# Dataset overrides
dataset_type = 'RaindropsOnWindshieldDataset'
data_root = 'data/raindrops_on_windshield'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img',
            seg_map_path='train/label'), # TODO
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img',
            seg_map_path='val/label'), # TODO
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=4,
#     dataset=dict(
#         data_root=data_root,
#         type=dataset_type,
#     )
# )
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     dataset=dict(
#         data_root=data_root,
#         type=dataset_type,
#     )
# )
# test_dataloader = val_dataloader
