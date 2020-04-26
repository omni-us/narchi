local input_size = std.extVar('input_size');
local nested3_size = std.extVar('nested3_size');
local output_size = std.extVar('output_size');

{
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'nested3',
            'path': 'nested3.jsonnet',
            'ext_vars': {
                'input_size': input_size,
                'output_size': nested3_size,
            },
        },
        {
            '_class': 'Linear',
            '_id': 'linear',
            'output_size': output_size,
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
