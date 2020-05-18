narchi_cli.py render image_gan.jsonnet;

narchi_cli.py render --ext_vars '{"image_channels": 3,
                                  "image_height": 48,
                                  "image_width": 32,
                                  "latent_size": 64,
                                  "hidden_sizes": [128, 256, 512]}' image_gan_generator.jsonnet;

narchi_cli.py render --ext_vars '{"image_channels": 3,
                                  "image_height": 48,
                                  "image_width": 32,
                                  "hidden_sizes": [512, 256, 128]}' image_gan_discriminator.jsonnet;
