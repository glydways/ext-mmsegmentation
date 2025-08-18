_base_ = [
    '../../_base_/models/fast_scnn.py', 'glyd_lens_artifact_1024x1024.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]

# Model override
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=1),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=1,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=1,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ]
)

# Training schedule overrides:
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(_delete_=True, max_norm=1.0)
)
train_cfg = dict(max_iters=160000, val_interval=1000) # run validation every 1000 steps


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
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
    )
)
test_dataloader = val_dataloader


