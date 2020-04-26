local input_size = std.extVar('input_size');
local output_size = std.extVar('output_size');

{
    'blocks': [
        {
            '_class': 'Linear',
            '_id': 'linear',
            'output_size': output_size,
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
