
local vocabulary_size = std.extVar('vocabulary_size');
local embedding_feats = 64;
local txt_layer_feats = [[32, 128], [64, 256], [64, 256]];
local img_layer_feats = [[16, 64], [32, 128], [64, 256]];
local shared_feats = 512;
local num_classes_task1 = 16;
local num_classes_task2 = 3;


local Fire(squeeze_feats, expand_feats, dims) = {
    '_class': 'Group',
    'blocks': [
        {
            '_class': 'Conv'+dims+'d',
            '_id': 'squeeze',
            'output_feats': squeeze_feats,
            'kernel_size': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'squeeze_activation',
            'inplace': true,
        },
        {
            '_class': 'Conv'+dims+'d',
            '_id': 'expand1',
            'output_feats': expand_feats,
            'kernel_size': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'expand1_activation',
            'inplace': true,
        },
        {
            '_class': 'Conv'+dims+'d',
            '_id': 'expand3',
            'output_feats': expand_feats,
            'kernel_size': 3,
            'padding': 1,
        },
        {
            '_class': 'ReLU',
            '_id': 'expand3_activation',
            'inplace': true,
        },
        {
            '_class': 'Concatenate',
            '_id': 'concat',
            'dim': 0,
        },
        {
            '_class': 'MaxPool'+dims+'d',
            '_id': 'maxpool',
            'kernel_size': 2,
            'stride': 2,
        },
    ],
    'graph': [
        'squeeze -> squeeze_activation',
        'squeeze_activation -> expand1 -> expand1_activation -> concat',
        'squeeze_activation -> expand3 -> expand3_activation -> concat',
        'concat -> maxpool',
    ],
    'input': 'squeeze',
    'output': 'maxpool',
};


{
    '_description': 'Simple text+image multimodal architecture for two document classification tasks.',
    'blocks': [
        {
            '_class': 'Sequential',
            '_id': 'text_features',
            'blocks': [
                {
                    '_class': 'Embedding',
                    'num_embeddings': vocabulary_size,
                    'output_feats': embedding_feats,
                },
                {
                    '_class': 'Reshape',
                    'reshape_spec': [1, 0],
                },
                Fire(txt_layer_feats[0][0], txt_layer_feats[0][1], dims=1),
                Fire(txt_layer_feats[1][0], txt_layer_feats[1][1], dims=1),
                Fire(txt_layer_feats[2][0], txt_layer_feats[2][1], dims=1),
                {
                    '_class': 'Reshape',
                    'reshape_spec': [1, 0],
                },
                {
                    '_class': 'LSTM',
                    'output_feats': 2*txt_layer_feats[2][1],
                    'num_layers': 2,
                    'bidirectional': true,
                },
                {
                    '_class': 'Reshape',
                    'reshape_spec': [1, 0],
                },
                {
                    '_class': 'AdaptiveAvgPool1d',
                    'output_feats': 1,
                },
                {
                    '_class': 'Reshape',
                    'reshape_spec': 'flatten',
                },
                {
                    '_class': 'Linear',
                    'output_feats': shared_feats,
                },
                {
                    '_class': 'ReLU',
                },
            ],
        },
        {
            '_class': 'Sequential',
            '_id': 'image_features',
            'blocks': [
                Fire(img_layer_feats[0][0], img_layer_feats[0][1], dims=2),
                Fire(img_layer_feats[1][0], img_layer_feats[1][1], dims=2),
                Fire(img_layer_feats[2][0], img_layer_feats[2][1], dims=2),
                {
                    '_class': 'AdaptiveAvgPool2d',
                    'output_feats': [1, 1],
                },
                {
                    '_class': 'Reshape',
                    'reshape_spec': 'flatten',
                },
                {
                    '_class': 'Linear',
                    'output_feats': shared_feats,
                },
                {
                    '_class': 'ReLU',
                },
            ],
        },
        {
            '_class': 'Add',
            '_id': 'add',
        },
        {
            '_class': 'Linear',
            '_id': 'fc_task1',
            'output_feats': num_classes_task1,
        },
        {
            '_class': 'Linear',
            '_id': 'fc_task2',
            'output_feats': num_classes_task2,
        },
    ],
    'graph': [
        'text -> text_features -> add',
        'image -> image_features -> add',
        'add -> fc_task1 -> logits_task1',
        'add -> fc_task2 -> logits_task2',
    ],
    'inputs': [
        {
            '_id': 'text',
            '_description': 'Sequence of indexes of text tokens extracted from a document image. Shape: LENGTH(variable).',
            '_shape': ['<<variable:L>>'],
        },
        {
            '_id': 'image',
            '_description': 'Image of a document. Shape: CHANNELS(fixed) × HEIGHT(variable) × WIDTH(variable).',
            '_shape': [3, '<<variable:H>>', '<<variable:W>>'],
        },
    ],
    'outputs': [
        {
            '_id': 'logits_task1',
            '_description': 'Class logits for first task.',
            '_shape': ['<<auto>>'],
        },
        {
            '_id': 'logits_task2',
            '_description': 'Class logits for second task.',
            '_shape': ['<<auto>>'],
        },
    ],
}
