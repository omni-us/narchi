local num_blocks = [2, 2, 2, 2];
local num_classes = 1000;

{
    '_description': 'Multi-scale ResNet with shared weights.',
    'blocks': [
        {
            '_class': 'AvgPool2d',
            '_id': 'downscale1',
            'kernel_size': 2,
            'stride': 2,
        },
        {
            '_class': 'AvgPool2d',
            '_id': 'downscale2',
            'kernel_size': 2,
            'stride': 2,
        },
        {
            '_class': 'Module',
            '_name': 'ResNet18',
            '_id': 'resnet1',
            '_id_share': 'resnet_share',
            '_path': 'resnet.jsonnet',
            '_ext_vars': {
                'num_blocks': num_blocks,
            },
        },
        {
            '_class': 'Module',
            '_name': 'ResNet18',
            '_id': 'resnet2',
            '_id_share': 'resnet_share',
            '_path': 'resnet.jsonnet',
            '_ext_vars': {
                'num_blocks': num_blocks,
            },
        },
        {
            '_class': 'Module',
            '_name': 'ResNet18',
            '_id': 'resnet3',
            '_id_share': 'resnet_share',
            '_path': 'resnet.jsonnet',
            '_ext_vars': {
                'num_blocks': num_blocks,
            },
        },
        {
            '_class': 'Add',
            '_id': 'merge',
        },
    ],
    'graph': [
        'image -> resnet1 -> merge',
        'image -> downscale1 -> resnet2 -> merge',
        'downscale1 -> downscale2 -> resnet3 -> merge',
        'merge -> logits',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_shape': [3, '<<variable:H>>', '<<variable:W>>'],
        },
    ],
    'outputs': [
        {
            '_id': 'logits',
            '_shape': [num_classes],
        },
    ],
}
