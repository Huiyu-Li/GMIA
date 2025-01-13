# Source: https://docs.libauc.org/_modules/libauc/models/densenet.html#DenseNet
# This implementation is adapted from https://github.com/pytorch/pytorch

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch import Tensor
from torch.jit.annotations import List


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Module):
    r"""
    Please refer to the source code for more details about this class.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, activation, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=0.01)),
        self.add_module('elu1', activation(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, momentum=0.01)),
        self.add_module('elu2', activation(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.elu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.elu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, activation, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                activation=activation,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, activation):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, momentum=0.01))
        self.add_module('ELU', activation(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""
        Densenet-BC model class, based on `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int): how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints): how many layers in each pooling block
            num_init_features (int): the number of filters to learn in the first convolution layer
            bn_size (int): multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float): dropout rate after each dense layer
            num_classes (int): number of classification classes
            memory_efficient (bool): If True, uses checkpointing. Much more memory efficient,
              but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, num_in_features=3, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, drop_rate=0.0, num_classes=1, memory_efficient=False, activations='ReLU', last_activation=None):

        super(DenseNet, self).__init__()
        self.step_counter = 0
        self.block_config = block_config

        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activations])

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_in_features, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features, momentum=0.01)),  #epsilon=0.001
            ('elu0', self.activation(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                activation=self.activation,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, 
                                    activation = self.activation)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features, momentum=0.01))

        # Linear layer
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(num_features, 512)),
            ('elu0', self.activation(inplace=True)),
            ('bn0', nn.BatchNorm1d(512, eps=1e-05)),
            ('fc1', nn.Linear(512, num_classes))
            ]))
        self.classifier.bn0.requires_grad = False

        self.last_activation = last_activation
        self.num_classes = num_classes
        if self.last_activation is not None:
            self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight) 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)

    def encoder(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        classifier_fc0 = getattr(self.classifier, 'fc0')
        classifier_elu0 = getattr(self.classifier, 'elu0')
        out = classifier_fc0(out)
        out = classifier_elu0(out)
        return out

    def forward(self, x: Tensor, mode: bool) -> Tensor:
        if mode == 'train':
            self.step_counter += 1
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        else:
            out = self.sigmoid(out)
        return out

def DenseNet121(num_in_features, pretrained=False, progress=True, activations='relu', **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    # print(f'kwargs: {kwargs}') # {'last_activation': None, 'num_classes': 4}
    model = DenseNet(num_in_features, num_init_features=64, growth_rate=32, 
                    block_config=(6, 12, 24, 16), activations=activations, **kwargs)
    
    if pretrained:
        _load_state_dict(model, model_urls['densenet121'], progress)
    return model

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'ELU.1', 'conv.1', 'norm.2', 'ELU.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|ELU|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    # pop the keys in pretrained state_dict
    state_dict.pop('classifier.weight', None)
    state_dict.pop('classifier.bias', None)     
    model.load_state_dict(state_dict, strict=False)