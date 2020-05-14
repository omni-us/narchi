local input_size = std.extVar('input_size');
local output_feats = std.extVar('output_feats');

{
    'blocks': [
        {
            '_class': 'Linear',
            '_id': 'linear',
            'output_feats': output_feats,
        },
        {
            '_class': 'Dropout',
            '_id': 'dropout',
        },
    ],
    'graph': [
        'input -> linear -> dropout -> output',
    ],
    'inputs': [
        {
            '_id': 'input',
            '_shape': [input_size],
        },
    ],
    'outputs': [
        {
            '_id': 'output',
            '_shape': ['<<auto>>'],
        },
    ],
}
