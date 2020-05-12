
local input_channels = 3;
local input_height = '<<variable:H>>';
local input_width = '<<variable:W>>';
local num_classes = 1000;


{
    '_description': 'ImageNet classifier.',
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'classifier',
            '_path': 'resnet.jsonnet',
            #'state_dict': 'resnet18-5c106cde.pth',
            #'state_dict': 'resnet34-333f7ec4.pth',
            '_ext_vars': {
                'num_blocks': [2, 2, 2, 2],  # resnet18
                #'num_blocks': [3, 4, 6, 3],  # resnet34
            },
        },
    ],
    'graph': [
        'image -> classifier -> classprob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_description': 'Input image.',
            '_shape': [input_channels, input_height, input_width],
        },
    ],
    'outputs': [
        {
            '_id': 'classprob',
            '_description': 'Class probabilities.',
            '_shape': [num_classes],
        },
    ],
}
