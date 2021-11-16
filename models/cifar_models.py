import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils


class ConvEncoder(nn.Module):
  def __init__(self):
    super(ConvEncoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
    self.relu1 = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
    self.relu3 = nn.ReLU(inplace=True)
    self.maxpool2 = nn.MaxPool2d(2)
    self.linear1 = nn.Linear(8 * 8 * 128, 512)
    self.relu4 = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.maxpool1(self.relu1(self.conv1(x)))
    out = self.relu2(self.conv2(out))
    out = self.maxpool2(self.relu3(self.conv3(out)))
    out = out.view(out.size(0), -1)
    out = self.relu4(self.linear1(out))
    return out


class SimpleDecoder(nn.Module):
  def __init__(self, num_outputs, num_features=512):
    super(SimpleDecoder, self).__init__()
    self.linear2 = nn.Linear(num_features, num_outputs)

  def forward(self, x):
    out = self.linear2(x)
    return out


class Block(nn.Module):

  def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
               activation=F.relu, downsample=False):
    super(Block, self).__init__()

    self.activation = activation
    self.downsample = downsample

    self.learnable_sc = (in_ch != out_ch) or downsample
    if h_ch is None:
      h_ch = in_ch
    else:
      h_ch = out_ch

    self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
    self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
    if self.learnable_sc:
      self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
    if self.learnable_sc:
      init.xavier_uniform_(self.c_sc.weight.data)

  def forward(self, x):
    return self.shortcut(x) + self.residual(x)

  def shortcut(self, x):
    if self.learnable_sc:
      x = self.c_sc(x)
    if self.downsample:
      return F.avg_pool2d(x, 2)
    return x

  def residual(self, x):
    h = self.c1(self.activation(x))
    h = self.c2(self.activation(h))
    if self.downsample:
      h = F.avg_pool2d(h, 2)
    return h


class OptimizedBlock(nn.Module):

  def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
    super(OptimizedBlock, self).__init__()
    self.activation = activation

    self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
    self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
    self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c_sc.weight.data)

  def forward(self, x):
    return self.shortcut(x) + self.residual(x)

  def shortcut(self, x):
    return self.c_sc(F.avg_pool2d(x, 2))

  def residual(self, x):
    h = self.activation(self.c1(x))
    return F.avg_pool2d(self.c2(h), 2)


class BlockNoSN(nn.Module):

  def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
               activation=F.relu, downsample=False):
    super(BlockNoSN, self).__init__()

    self.activation = activation
    self.downsample = downsample

    self.learnable_sc = (in_ch != out_ch) or downsample
    if h_ch is None:
      h_ch = in_ch
    else:
      h_ch = out_ch

    self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
    self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
    if self.learnable_sc:
      self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
    if self.learnable_sc:
      init.xavier_uniform_(self.c_sc.weight.data)

  def forward(self, x):
    return self.shortcut(x) + self.residual(x)

  def shortcut(self, x):
    if self.learnable_sc:
      x = self.c_sc(x)
    if self.downsample:
      return F.avg_pool2d(x, 2)
    return x

  def residual(self, x):
    h = self.c1(self.activation(x))
    h = self.c2(self.activation(h))
    if self.downsample:
      h = F.avg_pool2d(h, 2)
    return h


class OptimizedBlockNoSN(nn.Module):

  def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
    super(OptimizedBlockNoSN, self).__init__()
    self.activation = activation

    self.c1 = nn.Conv2d(in_ch, out_ch, ksize, 1, pad)
    self.c2 = nn.Conv2d(out_ch, out_ch, ksize, 1, pad)
    self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
    init.xavier_uniform_(self.c_sc.weight.data)

  def forward(self, x):
    return self.shortcut(x) + self.residual(x)

  def shortcut(self, x):
    return self.c_sc(F.avg_pool2d(x, 2))

  def residual(self, x):
    h = self.activation(self.c1(x))
    return F.avg_pool2d(self.c2(h), 2)


class SNPDCifar(nn.Module):

  def __init__(self, encoder, num_features, num_classes=0):
    super(SNPDCifar, self).__init__()
    self.encoder = encoder
    self.num_features = num_features
    self.num_classes = num_classes

  def forward(self, *input):
    raise NotImplementedError


class SNPDFC1NoBN(SNPDCifar):

  def __init__(self, encoder, num_features=512, num_classes=0):
    super(SNPDFC1NoBN, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None):
    h = self.encoder(x)
    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class SNPDFC3(SNPDCifar):

  def __init__(self, encoder, num_features=512, num_classes=0):
    super(SNPDFC3, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.linear1 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None, decoder_only=False):
    if decoder_only: 
      h = x
    else:
      h = self.encoder(x)
    h = self.linear1(h)
    h = self.relu1(h)
    h = self.linear2(h)
    h = self.relu2(h)

    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class SNPDFC3NoBNLogits(SNPDCifar):

  def __init__(self, encoder, decoder, num_features=512, num_classes=0):
    super(SNPDFC3NoBNLogits, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)
    self.decoder = decoder

    self.linear1 = utils.spectral_norm(nn.Linear(num_classes, num_features))
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None):
    h = self.decoder(self.encoder(x))
    h = self.linear1(h)
    h = self.relu1(h)
    h = self.linear2(h)
    h = self.relu2(h)

    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class SNPDSkipNoBN(SNPDCifar):

  def __init__(self, encoder, num_features=512, num_classes=0):
    super(SNPDSkipNoBN, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.linear1 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None):
    h_in = self.encoder(x)
    h = self.linear1(h_in)
    h = self.relu1(h)
    h = self.linear2(h)
    h = self.relu2(h)

    h = h + h_in
    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class SNPDResnet(SNPDCifar):

  def __init__(self, encoder, num_features, num_classes=0, activation=F.relu):
    super(SNPDResnet, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.activation = activation
    self.block1 = OptimizedBlock(num_features, num_features)
    self.block2 = Block(num_features, num_features * 2,
                        activation=activation, downsample=False)
    self.block3 = Block(num_features * 2, num_features * 4,
                        activation=activation, downsample=True)
    # self.block4 = Block(num_features * 4, num_features * 8,
    #                     activation=activation, downsample=True)
    self.l7 = utils.spectral_norm(nn.Linear(num_features * 4, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features * 4))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None):

    h = self.encoder(x)
    h = self.block1(h)
    h = self.block2(h)
    h = self.block3(h)
    # h = self.block4(h)
    h = self.activation(h)
    # Global pooling
    h = torch.sum(h, dim=(2, 3))
    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class SNPDResnetNoSN(SNPDCifar):

  def __init__(self, encoder, num_features, num_classes=0, activation=F.relu):
    super(SNPDResnetNoSN, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.activation = activation
    self.block1 = OptimizedBlockNoSN(num_features, num_features)
    self.block2 = BlockNoSN(num_features, num_features * 2,
                        activation=activation, downsample=False)
    self.block3 = BlockNoSN(num_features * 2, num_features * 4,
                        activation=activation, downsample=True)
    # self.block4 = BlockNoSN(num_features * 4, num_features * 8,
    #                     activation=activation, downsample=True)
    self.l7 = nn.Linear(num_features * 4, 1)
    if num_classes > 0:
      self.l_y = nn.Embedding(num_classes, num_features * 4)

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None):

    h = self.encoder(x)
    h = self.block1(h)
    h = self.block2(h)
    h = self.block3(h)
    # h = self.block4(h)
    h = self.activation(h)

    # Global pooling
    h = torch.sum(h, dim=(2, 3))
    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output


class CifarAdvDecoderConv3(nn.Module):
  def __init__(self, num_features):
    super(CifarAdvDecoderConv3, self).__init__()
    self.conv1 = nn.Conv2d(512, num_features, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU(inplace=False)
    self.conv2 = nn.Conv2d(num_features, num_features * 2, kernel_size=3)
    self.relu2 = nn.ReLU(inplace=False)
    self.linear3 = nn.Linear(num_features * 2, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = torch.sum(out, dim=(2, 3))
    out = self.linear3(out)
    out = self.sig(out)
    return out


class CifarAdvDecoderFC3LRelu(nn.Module):
  def __init__(self, num_features):
    super(CifarAdvDecoderFC3LRelu, self).__init__()
    self.linear1 = nn.Linear(num_features, num_features)
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = nn.Linear(num_features, num_features)
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear3 = nn.Linear(num_features, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    out = self.linear1(x)
    out = self.relu1(out)
    out = self.linear2(out)
    out = self.relu2(out)
    out = self.linear3(out)
    out = self.sig(out)
    return out


