
local input_channels = 3;
local input_height = 64;
local input_width = '<<variable:W>>';
local out_length = '<<variable:W/8>>';  # Make sure output length is 1/8 of input image width.
local num_symbols = std.extVar('num_symbols');
local conv1_features = 16;
local conv2_features = 16;
local conv3_features = 32;
local conv4_features = 32;
local lstm_layers = 3;
local lstm_features = 512;
local lstm_dropout = 0.5;
local lstm_bidirectional = true;


local Conv2dBlock(_id, output_feats, kernel_size=3, padding=1, leakyrelu=0.01, maxpool=true, maxpool_size=2) = {
    '_class': 'Sequential',
    '_id': _id,
    'blocks': std.prune([
        {
            '_class': 'Conv2d',
            'output_feats': output_feats,
            'kernel_size': kernel_size,
            'padding': padding,
        },
        {
            '_class': 'BatchNorm2d',
        },
        {
            '_class': 'LeakyReLU',
            'negative_slope': leakyrelu,
        },
        if maxpool then
        {
            '_class': 'MaxPool2d',
            'kernel_size': maxpool_size,
            'stride': maxpool_size,
        },
    ]),
};


{
    '_description': 'Default architecture from Laia for handwritten text recognition.',
    'blocks': [
        Conv2dBlock(_id='conv1', output_feats=conv1_features),
        Conv2dBlock(_id='conv2', output_feats=conv2_features),
        Conv2dBlock(_id='conv3', output_feats=conv3_features, maxpool=false),
        Conv2dBlock(_id='conv4', output_feats=conv4_features),
        {
            '_class': 'Reshape',
            '_id': 'to_1d',
            'reshape_spec': [2, [0, 1]],
        },
        {
            '_class': 'LSTM',
            '_id': 's3blstm',
            'output_feats': lstm_features,
            'num_layers': lstm_layers,
            'dropout': lstm_dropout,
            'bidirectional': lstm_bidirectional,
        },
        {
            '_class': 'Linear',
            '_id': 'fc',
            'output_feats': num_symbols,
        },
    ],
    'graph': [
        'image -> conv1 -> conv2 -> conv3 -> conv4 -> to_1d -> s3blstm -> fc -> symbprob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_description': 'Image of a single cropped line of text. Shape: CHANNELS(fixed) x HEIGHT(fixed) x WIDTH(variable).',
            '_shape': [input_channels, input_height, input_width],
        },
    ],
    'outputs': [
        {
            '_id': 'symbprob',
            '_description': 'Sequence of posterior probabilities for known symbols. Shape: SEQ_LENGTH(variable) x NUM_SYMBOLS(fixed).',
            '_shape': [out_length, num_symbols],
        },
    ],
}
