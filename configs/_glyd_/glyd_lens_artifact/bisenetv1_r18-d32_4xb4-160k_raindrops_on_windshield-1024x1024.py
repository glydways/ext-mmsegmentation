_base_ = [
    '../../_base_/models/bisenetv1_r18-d32.py',
    'glyd_lens_artifact_1024x1024.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

# Model overrides
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=1,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=1,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    ]
)

# Training schedule overrides
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(_delete_=True, max_norm=1.0)
)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=5000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=5000,
        end=160000,
        by_epoch=False,
    )
]
train_cfg = dict(max_iters=160000, val_interval=1000) # run validation every 1000 steps

# Runtime overrides
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend')
                ]
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
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
    )
)
test_dataloader = val_dataloader