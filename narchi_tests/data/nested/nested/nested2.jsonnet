local input_size = std.extVar('input_size');
local nested3_size = std.extVar('nested3_size');
local output_feats = std.extVar('output_feats');

{
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'nested3',
            '_path': '../nested3.jsonnet',
            '_ext_vars': {
                'input_size': input_size,
                'output_feats': nested3_size,
            },
        },
        {
            '_class': 'Linear',
            '_id': 'linear',
            'output_feats': output_feats,
        },
        {
            '_class': 'ReLU',
            '_id': 'relu',
        },
    ],
    'graph': [
        'input -> nested3 -> linear -> relu -> output',
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
