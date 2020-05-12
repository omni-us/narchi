local image_channels = 3;
local image_width = 32;
local image_height = 48;
local latent_size = 64;
local generator_hidden_sizes = [128, 256, 512];
local discriminator_hidden_sizes = [512, 256, 128];


{
    '_description': 'Simple GAN architecture for generating images.',
    'blocks': [
        {
            '_class': 'Module',
            '_id': 'generator',
            '_name': 'Generator',
            '_path': 'image_gan_generator.jsonnet',
            '_ext_vars': {
                'image_channels': image_channels,
                'image_height': image_height,
                'image_width': image_width,
                'latent_size': latent_size,
                'hidden_sizes': generator_hidden_sizes,
            },
        },
        {
            '_class': 'Module',
            '_id': 'discriminator',
            '_name': 'Discriminator',
            '_path': 'image_gan_discriminator.jsonnet',
            '_ext_vars': {
                'image_channels': image_channels,
                'image_height': image_height,
                'image_width': image_width,
                'hidden_sizes': discriminator_hidden_sizes,
            },
        },
    ],
    'graph': [
        'latent -> generator -> discriminator -> prob',
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
            '_id': 'prob',
            '_description': 'Probability that the image is real.',
            '_shape': [1],
        },
    ],
}
