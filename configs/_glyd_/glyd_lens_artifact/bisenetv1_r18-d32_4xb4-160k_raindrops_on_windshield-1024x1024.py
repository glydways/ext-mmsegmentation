_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    'glyd_lens_artifact_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Model overrides
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=[
        dict(num_classes=2),
        dict(num_classes=2)
    ]
)

# Training schedule overrides
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
train_cfg = dict(max_iters=160000, val_interval=1000) # run validation every 1000 steps

# Runtime overrides
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]

# Dataset overrides
dataset_type = 'RaindropsOnWindshieldDataset'
data_root = 'data/raindrops_on_windshield'
crop_size = (1024, 1024)
