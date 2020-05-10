local input_size = 128;
local hidden_size = std.extVar('hidden_size');
local output_size = std.extVar('output_size');

{
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'nested2',
            'path': 'nested2.jsonnet',
            'ext_vars': {
                'input_size': input_size,
                'nested3_size': hidden_size,
            },
        },
        {
            '_class': 'Softmax',
            '_id': 'softmax',
        },
    ],
    'graph': [
        'input -> nested2 -> softmax -> output',
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
            '_shape': [output_size],
        },
    ],
}
