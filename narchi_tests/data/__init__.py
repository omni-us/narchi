import os

data_dir = os.path.dirname(os.path.realpath(__file__))

resnet_jsonnet = os.path.join(data_dir, 'resnet.jsonnet')
resnet_ext_vars = {'num_blocks': [2, 2, 2, 2]}

squeezenet_jsonnet = os.path.join(data_dir, 'squeezenet.jsonnet')

text_image_jsonnet = os.path.join(data_dir, 'text_image_classification.jsonnet')
text_image_ext_vars = {'vocabulary_size': 5000}

laia_jsonnet = os.path.join(data_dir, 'laia.jsonnet')
laia_ext_vars = {'num_symbols': 68}
laia_shapes = [[16, 32, '<<variable:W/2>>'],
               [16, 16, '<<variable:W/4>>'],
               [32, 16, '<<variable:W/4>>'],
               [32, 8, '<<variable:W/8>>'],
               ['<<variable:W/8>>', 256],
               ['<<variable:W/8>>', 512],
               ['<<variable:W/8>>', 68]]

nested1_jsonnet = os.path.join(data_dir, 'nested1.jsonnet')
nested2_jsonnet = os.path.join(data_dir, 'nested/nested/nested2.jsonnet')
nested3_jsonnet = os.path.join(data_dir, 'nested/nested3.jsonnet')
nested1_ext_vars = {'hidden_size': 64, 'output_feats': 16}
nested2_ext_vars = {'input_size': 128, 'nested3_size': 64, 'output_feats': 16}
nested3_ext_vars = {'input_size': 64, 'output_feats': 16}
