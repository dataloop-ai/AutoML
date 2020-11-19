from .retinanet import retinanet
from .resnet import ResNet
from .pyramidnet import PyramidNet
from .shakeshake.shake_resnet import ShakeResNet
from .wideresnet import WideResNet
from .shakeshake.shake_resnext import ShakeResNeXt
from .efficientnet_pytorch import EfficientNet, RoutingFn
# from tf_port.tpu_bn import TpuBatchNormalization


def get_model(model_name, num_classes=10, backbone_depth=None, ratios=None, scales=None, weights_dir=None,
              pretrained=True):

    networks = {'retinanet': retinanet, 'wideresnet': WideResNet, 'pyramid': PyramidNet, 'efficientnet': EfficientNet}
    model = networks[model_name](depth=backbone_depth, num_classes=num_classes, ratios=ratios, scales=scales,
                                 weights_dir=weights_dir,
                                 pretrained=pretrained)

    # THIS CAN SUBSTANTIALLY SLOW DOWN THE FIRST EPOCH WHEN TIMES ARE DIFFERENT
    # cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
