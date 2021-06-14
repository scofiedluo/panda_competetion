CATE_ID = '1'

classes_dict = {'1': 'visible body', '2': 'full body', '3': 'head', '4': 'vehicle'}
json_pre_dict = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4':'vehicle'}

data_root = '/lustre/home/acct-eedxw/eedxw-user3/tianchi_PANDA/PANDA-Toolkit/PANDA_COCO_TYPE_DATA/split_' + json_pre_dict[CATE_ID].split('_')[0] +'_train/'
anno_root = '/lustre/home/acct-eedxw/eedxw-user3/tianchi_PANDA/PANDA-Toolkit/PANDA_COCO_TYPE_DATA/coco_format_json'

classes = (classes_dict[CATE_ID],)
json_pre = json_pre_dict[CATE_ID]

dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_train.json',
        img_prefix=data_root + 'image_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# We can use the pre-trained Faster RCNN model to obtain higher performance
# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'