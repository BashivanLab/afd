import torch
import torch.nn as nn
from torch.nn import utils
from torch.nn import init
import torch.nn.functional as F


class LeNetFeats(nn.Module):
  def __init__(self, normalize=False):
    super(LeNetFeats, self).__init__()
    self.normalize = normalize
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
    self.relu1 = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
    self.relu2 = nn.ReLU(inplace=True)
    self.maxpool2 = nn.MaxPool2d(2)
    self.linear1 = nn.Linear(7 * 7 * 64, 64)
    self.relu3 = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.maxpool1(self.relu1(self.conv1(x)))
    out = self.maxpool2(self.relu2(self.conv2(out)))
    out = out.view(out.size(0), -1)
    out = self.relu3(self.linear1(out))
    if self.normalize: 
      out = F.normalize(out)
    return out


class LeNetFeatsNorm(LeNetFeats):
  def __init__(self):
    super(LeNetFeatsNorm, self).__init__(normalize=True)
    

class LeNetDecoder(nn.Module):
  def __init__(self, num_outputs, num_features=64):
    super(LeNetDecoder, self).__init__()
    self.linear2 = nn.Linear(num_features, num_outputs)

  def forward(self, x):
    out = self.linear2(x)
    return out


class SNPDMNIST(nn.Module):

  def __init__(self, encoder, num_features, num_classes=0):
    super(SNPDMNIST, self).__init__()
    self.encoder = encoder
    self.num_features = num_features
    self.num_classes = num_classes

  def forward(self, *input):
    raise NotImplementedError


class SNPDFC3(SNPDMNIST):

  def __init__(self, encoder, num_features, num_classes=0, activation=F.relu):
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

