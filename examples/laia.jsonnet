
local num_symbols = std.extVar('num_symbols');
local out_length = '<<variable:W/8>>';  # Make sure output length is 1/8 of input image width.
#local out_length = '<<auto>>';
local input_channels = 3;
local input_height = 64;
local conv1_features = 16;
local conv2_features = 16;
local conv3_features = 32;
local conv4_features = 32;
local lstm_layers = 3;
local lstm_features = 512;
local lstm_dropout = 0.5;
local lstm_bidirectional = true;


local Conv2dBlock(_id, out_features, kernel_size=3, padding=1, leakyrelu=0.01, maxpool=true, maxpool_size=2) = {
    '_class': 'Sequential',
    '_id': _id,
    'blocks': std.prune([
        {
            '_class': 'Conv2d',
            'out_features': out_features,
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
        Conv2dBlock(_id='conv1', out_features=conv1_features),
        Conv2dBlock(_id='conv2', out_features=conv2_features),
        Conv2dBlock(_id='conv3', out_features=conv3_features, maxpool=false),
        Conv2dBlock(_id='conv4', out_features=conv4_features),
        {
            '_class': 'Reshape2dTo1d',
            '_id': 'to_1d',
        },
        {
            '_class': 'LSTM',
            '_id': 's3blstm',
            'out_features': lstm_features,
            'num_layers': lstm_layers,
            'dropout': lstm_dropout,
            'bidirectional': lstm_bidirectional,
        },
        {
            '_class': 'Linear',
            '_id': 'fc',
        },
    ],
    'graph': [
        'image -> conv1 -> conv2 -> conv3 -> conv4 -> to_1d -> s3blstm -> fc -> symbprob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_description': 'Image of a single cropped line of text. Shape: CHANNELS(fixed) x HEIGHT(fixed) x WIDTH(variable).',
            '_shape': [input_channels, input_height, '<<variable:W>>'],
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
