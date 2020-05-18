#local image_channels = 3;
#local image_height = 48;
#local image_width = 32;
#local latent_size = 64;
#local hidden_sizes = [128, 256, 512];

local image_channels = std.extVar('image_channels');
local image_height = std.extVar('image_height');
local image_width = std.extVar('image_width');
local latent_size = std.extVar('latent_size');
local hidden_sizes = std.extVar('hidden_sizes');


local HiddenBlock(n) = {
    '_class': 'Sequential',
    '_id': 'hidden'+(n+1),
    'blocks': [
        {
            '_class': 'Linear',
            'output_feats': hidden_sizes[n],
        },
        {
            '_class': 'LeakyReLU',
            'negative_slope': 0.1,
        },
    ],
};


{
    '_description': 'Generator module for a simple image GAN architecture.',
    'blocks': [HiddenBlock(n) for n in std.range(0, std.length(hidden_sizes)-1)]+[
        {
            '_class': 'Linear',
            '_id': 'image_size',
            'output_feats': image_channels*image_height*image_width,
        },
        {
            '_class': 'Sigmoid',
            '_id': 'image_range',
        },
        {
            '_class': 'Reshape',
            '_id': 'image_shape',
            'reshape_spec': [{'0': [image_channels, image_height, image_width]}],
        },
    ],
    'graph': [
        'latent -> hidden1',
    ]+[
        'hidden'+n+' -> hidden'+(n+1) for n in std.range(1, std.length(hidden_sizes)-1)
    ]+[
        'hidden'+std.length(hidden_sizes)+' -> image_size -> image_range -> image_shape -> generated',
    ],
    'inputs': [
        {
            '_id': 'latent',
            '_description': 'Latent vector.',
            '_shape': [latent_size],
        },
    ],
    'outputs': [
        {
            '_id': 'generated',
            '_description': 'The generated image with shape C×H×W and in range [0,1].',
            '_shape': [image_channels, image_height, image_width],
        },
    ],
}
