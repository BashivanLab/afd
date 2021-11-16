'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, activation=F.relu):
    super(BasicBlock, self).__init__()
    self.activation = activation
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = self.activation(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.activation(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, activation=F.relu):
    super(Bottleneck, self).__init__()
    self.activation = activation
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = self.activation(self.bn1(self.conv1(x)))
    out = self.activation(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = self.activation(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10, activation=F.relu,
               output_features=False, normalize=False, base_feats=16):
    super(ResNet, self).__init__()
    self.activation = activation
    self.output_features = output_features
    self.normalize = normalize
    self.in_planes = 16

    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.layers = OrderedDict()
    for idx, n in enumerate(num_blocks):
      self.layers[f'{idx+1}'] = self._make_layer(block, base_feats * (idx+1), n, stride=2, activation=self.activation)
    self.layers = torch.nn.Sequential(self.layers)
    self.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    if not self.output_features:
      self.linear = nn.Linear(base_feats * len(num_blocks) * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride, activation):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, activation))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.activation(self.bn1(self.conv1(x)))
    out = self.layers(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    if self.normalize:
      out = F.normalize(out)

    if not self.output_features:
      out = self.linear(out)

    return out


def ResNet18Feats():
  return ResNet(BasicBlock, [2, 2, 2, 2], base_feats=16, output_features=True)


def ResNet18FeatsNorm():
  return ResNet(BasicBlock, [2, 2, 2, 2], base_feats=16, output_features=True, normalize=True)


def ResNet18FeatsNormLeaky():
  return ResNet(BasicBlock, [2, 2, 2, 2], base_feats=16, output_features=True, normalize=True, activation=F.leaky_relu)
  

def ResNet18FeatsNormSILU(): 
  return ResNet(BasicBlock, [2, 2, 2, 2], base_feats=16, output_features=True, 
  normalize=True, activation=F.silu)
