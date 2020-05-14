local input_size = 128;
local hidden_size = std.extVar('hidden_size');
local output_feats = std.extVar('output_feats');

{
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'nested2',
            '_path': 'nested/nested/nested2.jsonnet',
            '_ext_vars': {
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
            '_shape': [output_feats],
        },
    ],
}
