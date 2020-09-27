
narchi_cli.py render --nested_depth 1 imagenet_classifier.jsonnet imagenet_classifier_resnet_depth1.pdf
narchi_cli.py render --nested_depth 2 imagenet_classifier.jsonnet imagenet_classifier_resnet_depth2.pdf
narchi_cli.py render --nested_depth 3 imagenet_classifier.jsonnet imagenet_classifier_resnet_depth3.pdf
narchi_cli.py render --nested_depth 4 imagenet_classifier.jsonnet imagenet_classifier_resnet_depth4.pdf
narchi_cli.py render --nested_depth 5 imagenet_classifier.jsonnet imagenet_classifier_resnet_depth5.pdf

narchi_cli.py render --ext_vars '{"num_blocks": [2, 2, 2, 2]}' resnet.jsonnet resnet18.pdf;
narchi_cli.py render --ext_vars '{"num_blocks": [3, 4, 6, 3]}' resnet.jsonnet resnet34.pdf;

narchi_cli.py render --nested_depth 2 resnet_multiscale.jsonnet;

narchi_cli.py render squeezenet.jsonnet;
