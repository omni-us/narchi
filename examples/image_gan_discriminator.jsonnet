#local image_channels = 3;
#local image_height = 48;
#local image_width = 32;
#local hidden_sizes = [512, 256, 128];

local image_channels = std.extVar('image_channels');
local image_height = std.extVar('image_height');
local image_width = std.extVar('image_width');
local hidden_sizes = std.extVar('hidden_sizes');


local HiddenBlock(n) = {
    '_class': 'Sequential',
    '_id': 'hidden'+(n+1),
    'blocks': [
        {
            '_class': 'Linear',
            'output_size': hidden_sizes[n],
        },
        {
            '_class': 'LeakyReLU',
            'negative_slope': 0.1,
        },
    ],
};


{
    '_description': 'Discriminator module for a simple image GAN architecture.',
    'blocks': [
        {
            '_class': 'Reshape',
            '_id': 'flatten',
            'reshape_spec': [[0, 1, 2]],
        },
    ]+[HiddenBlock(n) for n in std.range(0, std.length(hidden_sizes)-1)]+[
        {
            '_class': 'Linear',
            '_id': 'linear',
            'output_size': 1,
        },
        {
            '_class': 'Sigmoid',
            '_id': 'range',
        },
    ],
    'graph': [
        'image -> flatten -> hidden1',
    ]+[
        'hidden'+n+' -> hidden'+(n+1) for n in std.range(1, std.length(hidden_sizes)-1)
    ]+[
        'hidden'+std.length(hidden_sizes)+' -> linear -> range -> prob',
    ],
    'inputs': [
        {
            '_id': 'image',
            '_description': 'Real or generated image with shape CxHxW and range [0,1].',
            '_shape': [image_channels, image_height, image_width],
        },
    ],
    'outputs': [
        {
            '_id': 'prob',
            '_description': 'Probability that the image is real.',
            '_shape': [1],
        },
    ],
}
