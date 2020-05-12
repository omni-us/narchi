#local num_blocks = [2, 2, 2, 2];  # resnet18 -> parse as: jsonnet --ext-code 'num_blocks=[2, 2, 2, 2]' resnet.jsonnet
#local num_blocks = [3, 4, 6, 3];  # resnet34 -> parse as: jsonnet --ext-code 'num_blocks=[3, 4, 6, 3]' resnet.jsonnet
local num_blocks = std.extVar('num_blocks');
local layer_output_size = [64, 128, 256, 512];
local num_classes = 1000;


local Conv3x3(_id, output_size, stride=1) = {
    '_class': 'Conv2d',
    '_id': _id,
    'output_size': output_size,
    'kernel_size': 3,
    'padding': 1,
    'stride': stride,
    'dilation': 1,
    'bias': false,
};


local Downsample(output_size) = {
    '_class': 'Sequential',
    '_id': 'downsample',
    'blocks': [
        {
            '_class': 'Conv2d',
            'output_size': output_size,
            'kernel_size': 1,
            'padding': 0,
            'stride': 2,
            'bias': false,
        },
        {
            '_class': 'BatchNorm2d',
        },
    ],
};


local ResBlock(n, output_size, downsample) = {
    local stride = if downsample && n == 0 then 2 else 1,
    '_class': 'Group',
    '_name': 'ResBlock',
    'blocks': std.prune([
        {
            '_class': 'Identity',
            '_id': 'ident',
        },
        Conv3x3(_id='conv1', output_size=output_size, stride=stride),
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
        if downsample && n == 0 then
        Downsample(output_size),
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
        'ident -> conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> add -> relu2',
        if downsample && n == 0 then
        'ident -> downsample -> add'
        else
        'ident -> add',
    ],
    'input': 'ident',
    'output': 'relu2',
};


local MakeLayer(num, downsample) = {
    '_class': 'Sequential',
    '_id': 'layer'+(num+1),
    'blocks': [ResBlock(n=n, output_size=layer_output_size[num], downsample=downsample) for n in std.range(0, num_blocks[num]-1)],
};


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
            'bias': false,
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
        MakeLayer(num=0, downsample=false),
        MakeLayer(num=1, downsample=true),
        MakeLayer(num=2, downsample=true),
        MakeLayer(num=3, downsample=true),
        {
            '_class': 'AdaptiveAvgPool2d',
            '_id': 'avgpool',
            'output_size': [1, 1],
        },
        {
            '_class': 'Reshape',
            '_id': 'flatten',
            'reshape_spec': [[0, 1, 2]],
        },
        {
            '_class': 'Linear',
            '_id': 'fc',
            'output_size': num_classes,
        },
    ],
    'graph': [
        'image -> conv1 -> bn1 -> relu -> maxpool',
        'maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool',
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
