local num_classes = 1000;
#local num_layers = [2, 2, 2, 2];  # resnet18 -> parse as: jsonnet --ext-code 'num_layers=[2, 2, 2, 2]' resnet.jsonnet
#local num_layers = [3, 4, 6, 3];  # resnet34 -> parse as: jsonnet --ext-code 'num_layers=[3, 4, 6, 3]' resnet.jsonnet
local num_layers = std.extVar('num_layers');


local Conv3x3(_id, output_size) = {
    '_class': 'Conv2d',
    '_id': _id,
    'output_size': output_size,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
    'dilation': 1,
    'bias': false,
};


local Conv1x1(_id, output_size) = {
    '_class': 'Conv2d',
    '_id': _id,
    'output_size': output_size,
    'kernel_size': 1,
    'padding': 0,
    'stride': 2,
    'bias': false,
};


local MakeLayer(_id, n, output_size, downsample=false) = {
    '_class': 'Group',
    '_id': _id+'_'+n,
    '_name': 'ResBlock',
    'blocks': std.prune([
        if ! ( downsample && n == 0 ) then
        {
            '_class': 'Identity',
            '_id': 'ident',
        },
        if downsample && n == 0 then
        Conv1x1(_id='downsample_0', output_size=output_size),
        if downsample && n == 0 then
        {
            '_class': 'BatchNorm2d',
            '_id': 'downsample_1',
        },
        Conv3x3(_id='conv1', output_size=output_size),
        {
            '_class': 'BatchNorm2d',
            '_id': 'bn1',
        },
        {
            '_class': 'ReLU',
            '_id': 'relu1',
        },
        Conv3x3(_id='conv2', output_size=output_size),
        {
            '_class': 'BatchNorm2d',
            '_id': 'bn2',
        },
        {
            '_class': 'Add',
            '_id': 'add',
        },
        {
            '_class': 'ReLU',
            '_id': 'relu2',
        },
    ]),
    'graph': [
        if downsample && n == 0 then
        'downsample_0 -> downsample_1 -> conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> add -> relu2'
        else
        'ident -> conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> add -> relu2',
        if downsample && n == 0 then
        'downsample_1 -> add'
        else
        'ident -> add',
    ],    
    'input': if downsample && n == 0 then 'downsample_0' else 'ident',
    'output': 'relu2',
};


local ConnectLayer(_id, layers, from_node='', to_node='') =
    (if from_node != '' then [from_node+' -> '+_id+'_0'] else []) +
    ([_id+'_'+n+' -> '+_id+'_'+(n+1) for n in std.range(0, layers-2)]) +
    (if to_node != '' then [_id+'_'+(layers-1)+' -> '+to_node] else []);


{
    '_description': 'ResNet model from "Deep Residual Learning for Image Recognition" &lt;https://arxiv.org/pdf/1512.03385.pdf&gt;.',
    'blocks': [
        {
            '_class': 'Conv2d',
            '_id': 'conv1',
            'output_size': 64,
            'kernel_size': 7,
            'padding': 3,
            'stride': 2,
        },
        {
            '_class': 'BatchNorm2d',
            '_id': 'bn1',
        },
        {
            '_class': 'ReLU',
            '_id': 'relu',
        },
        {
            '_class': 'MaxPool2d',
            '_id': 'maxpool',
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
        },
    ]
    +[MakeLayer(_id='layer1', n=n, output_size=64, downsample=false) for n in std.range(0, num_layers[0]-1)]
    +[MakeLayer(_id='layer2', n=n, output_size=128, downsample=true) for n in std.range(0, num_layers[1]-1)]
    +[MakeLayer(_id='layer3', n=n, output_size=256, downsample=true) for n in std.range(0, num_layers[2]-1)]
    +[MakeLayer(_id='layer4', n=n, output_size=512, downsample=true) for n in std.range(0, num_layers[3]-1)]
    +[
        {
            '_class': 'AdaptiveAvgPool2d',
            '_id': 'avgpool',
            'output_size': [1, 1],
        },
        {
            '_class': 'Reshape',
            '_id': 'flatten',
            'output_shape': [[0, 1, 2]],
        },
        {
            '_class': 'Linear',
            '_id': 'fc',
            'output_size': num_classes,
        },
    ],
    'graph': [
        'image -> conv1 -> bn1 -> relu -> maxpool',
    ]
    +ConnectLayer(_id='layer1', layers=num_layers[0], from_node='maxpool')
    +ConnectLayer(_id='layer2', layers=num_layers[1], from_node='layer1_'+(num_layers[0]-1))
    +ConnectLayer(_id='layer3', layers=num_layers[2], from_node='layer2_'+(num_layers[1]-1))
    +ConnectLayer(_id='layer4', layers=num_layers[3], from_node='layer3_'+(num_layers[2]-1), to_node='avgpool')
    +[
        'avgpool -> flatten -> fc -> classprob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_shape': [3, '<<variable:H>>', '<<variable:W>>'],
        },
    ],
    'outputs': [
        {
            '_id': 'classprob',
            '_shape': [num_classes],
        },
    ],
}
