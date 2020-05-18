local num_classes = 1000;


local Fire(squeeze_planes, expand1x1_planes, expand3x3_planes) = {
    '_class': 'Group',
    '_name': 'FireBlock',
    'blocks': std.prune([
        {
            '_class': 'Conv2d',
            '_id': 'squeeze',
            'output_feats': squeeze_planes,
            'kernel_size': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'squeeze_activation',
            'inplace': true,
        },
        {
            '_class': 'Conv2d',
            '_id': 'expand1x1',
            'output_feats': expand1x1_planes,
            'kernel_size': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'expand1x1_activation',
            'inplace': true,
        },
        {
            '_class': 'Conv2d',
            '_id': 'expand3x3',
            'output_feats': expand3x3_planes,
            'kernel_size': 3,
            'padding': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'expand3x3_activation',
            'inplace': true,
        },
        {
            '_class': 'Concatenate',
            '_id': 'concat',
            'dim': 0,
        },
    ]),
    'graph': [
        'squeeze -> squeeze_activation',
        'squeeze_activation -> expand1x1 -> expand1x1_activation -> concat',
        'squeeze_activation -> expand3x3 -> expand3x3_activation -> concat',
    ],
    'input': 'squeeze',
    'output': 'concat',
};


{
    '_description': 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size &lt;https://arxiv.org/abs/1602.07360&gt;.',
    'blocks': [
        {
            '_class': 'Sequential',
            '_id': 'features',
            'blocks': [
                {
                    '_class': 'Conv2d',
                    'output_feats': 96,
                    'kernel_size': 7,
                    'stride': 2,
                    'padding': 3,  # modification
                },
                {
                    '_class': 'ReLU',
                    'inplace': true,
                },
                {
                    '_class': 'MaxPool2d',
                    'kernel_size': 3,
                    'stride': 2,
                    'ceil_mode': true,  # not yet supported
                    'padding': 1,  # modification
                },
                Fire(16, 64, 64),
                Fire(16, 64, 64),
                Fire(32, 128, 128),
                {
                    '_class': 'MaxPool2d',
                    'kernel_size': 3,
                    'stride': 2,
                    'ceil_mode': true,  # not yet supported
                    'padding': 1,  # modification
                },
                Fire(32, 128, 128),
                Fire(48, 192, 192),
                Fire(48, 192, 192),
                Fire(64, 128, 128),
                {
                    '_class': 'MaxPool2d',
                    'kernel_size': 3,
                    'stride': 2,
                    'ceil_mode': true,  # not yet supported
                    'padding': 1,  # modification
                },
                Fire(64, 256, 256),
            ],
        },
        {
            '_class': 'Sequential',
            '_id': 'classifier',
            'blocks': [
                {
                    '_class': 'Dropout',
                    'p': 0.5,
                },
                {
                    '_class': 'Conv2d',
                    'output_feats': num_classes,
                    'kernel_size': 1,
                },
                {
                    '_class': 'ReLU',
                    'inplace': true,
                },
                {
                    '_class': 'AdaptiveAvgPool2d',
                    '_id': 'avgpool',
                    'output_feats': [1, 1],
                },
                {
                    '_class': 'Reshape',
                    '_id': 'flatten',
                    'reshape_spec': 'flatten',
                },
            ],
        },
    ],
    'graph': [
        'image -> features -> classifier -> classprob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_shape': [3, '<<variable:H>>', '<<variable:W>>'],
        },
    ],
    'outputs': [
        {
            '_id': 'classprob',
            '_shape': [num_classes],
        },
    ],
}
