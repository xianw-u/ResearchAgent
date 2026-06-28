dataset_type = 'SUIMDataset'
data_root = 'data/suim'
crop_size = (512, 512)

model = dict(
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
)

optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01)
param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=1.0, begin=0, end=80000, by_epoch=False)
]
batch_size = 4
load_from = 'pretrained/mit_b0.pth'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize'),
    dict(type='Pad'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='Normalize'),
]
