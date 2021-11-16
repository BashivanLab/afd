import os
from filelock import FileLock
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack
from attacks.deepfool import DeepfoolLinfAttack
from autoattack import AutoAttack

import models.mnist_models as mnist_models
import models.resnet_mnist as resnet_mnist

import models.cifar_models as cifar_models
import models.resnet_cifar as resnet_cifar


def prepare_dirs(save_path, dataset, exp_name): 
  ROOT_PATH = save_path
  TRAINED_MODEL_PATH = os.path.join(ROOT_PATH, f'trained_models/{dataset}', exp_name)
  DATA_PATH = os.path.join(ROOT_PATH, 'data', dataset)

  postfix = 1
  safe_path = TRAINED_MODEL_PATH
  while os.path.exists(safe_path):
    safe_path = TRAINED_MODEL_PATH + f'_{postfix}'
    postfix += 1
  TRAINED_MODEL_PATH = safe_path
  os.makedirs(TRAINED_MODEL_PATH)
  return TRAINED_MODEL_PATH, DATA_PATH
  

def get_data_loaders(dataset, train_batch_size, test_batch_size, data_path, norm=False):
  if dataset == 'mnist':
    train_loader = get_mnist_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True)
    test_loader = get_mnist_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False)
  elif dataset == 'cifar10':
    train_loader = get_cifar10_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True, norm=norm)
    test_loader = get_cifar10_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False, norm=norm)
  elif dataset == 'cifar100':
    train_loader = get_cifar100_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True, norm=norm)
    test_loader = get_cifar100_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False, norm=norm)
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')
  return train_loader, test_loader


def get_attack(model, attack_name, dataset):
  if dataset == 'mnist':
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    elif attack_name == 'linf_deepfool':
      return DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=40, eps=0.3, clip_min=0.0, clip_max=1.0)

    elif attack_name == 'cw':
      return CarliniWagnerL2Attack(
        model, num_classes=10, max_iterations=20, learning_rate=0.1,
        clip_min=0.0, clip_max=1.0)
    elif attack_name == 'aa_apgdt': 
      aa = AA(model, norm='Linf', eps=0.3, n_iter=20, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-t']
      return aa
    elif attack_name == 'aa_apgdce': 
      aa = AA(model, norm='Linf', eps=0.3, n_iter=40, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-ce']
      return aa
    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')

  elif 'cifar' in dataset:
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
        nb_iter=20, eps_iter=2. / 255, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)
    elif attack_name == 'kl_linf_pgd': 
      return KLPGDAttack(
        model, eps=8. / 255,
        nb_iter=20, eps_iter=2. / 255, clip_min=0., clip_max=1.0, distance='l_inf')
    elif attack_name == 'aa_apgdce':
      aa = AA(model, norm='Linf', eps=8./255, n_iter=20, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-ce']
      return aa
    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')

  else:
    raise NotImplementedError(f'Dataset not recognized ({dataset})')


def get_model(dataset, model_name, num_decoder_features=512):
  if dataset == 'mnist':
    if model_name == 'lenet': 
      return mnist_models.LeNetFeats(), mnist_models.LeNetDecoder(10, num_features=64)
    elif model_name == 'lenetnorm': 
      return mnist_models.LeNetFeats(normalize=True), mnist_models.LeNetDecoder(10, num_features=64)      
    elif model_name == 'resnet18':
      return resnet_mnist.ResNet18Feats(), mnist_models.LeNetDecoder(10, num_features=64)
    elif model_name == 'resnet18norm':
      return resnet_mnist.ResNet18FeatsNorm(), mnist_models.LeNetDecoder(10, num_features=64)
    elif model_name == 'resnet18normleaky':
      return resnet_mnist.ResNet18FeatsNormLeaky(), mnist_models.LeNetDecoder(10, num_features=64)
    elif model_name == 'resnet18normsilu':
      return resnet_mnist.ResNet18FeatsNormSILU(), mnist_models.LeNetDecoder(10, num_features=64)
    else:
      raise ValueError(f'Model name not recognized ({model_name})')
  elif 'cifar' in dataset:
    if dataset == 'cifar10':
      num_classes = 10
    else:
      num_classes = 100
    if model_name == 'resnet18':
      return resnet_cifar.ResNet18Feats(), \
             resnet_cifar.ResNetDecoder(num_features=num_decoder_features, num_classes=num_classes)
    elif model_name == 'resnet18norm':
      return resnet_cifar.ResNet18FeatsNorm(), \
             resnet_cifar.ResNetDecoder(num_features=num_decoder_features, num_classes=num_classes)
    elif model_name == 'resnet18normleaky':
      return resnet_cifar.ResNet18FeatsNormLeaky(), \
             resnet_cifar.ResNetDecoder(num_features=num_decoder_features, num_classes=num_classes)
    elif model_name == 'resnet18normsilu':
      return resnet_cifar.ResNet18FeatsNormSILU(), \
             resnet_cifar.ResNetDecoder(num_features=num_decoder_features, num_classes=num_classes)
    else:
      raise ValueError(f'Model name not recognized ({model_name})')
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')


def get_decoder_class(dataset, decoder_name):
  if dataset == 'mnist':
    if decoder_name == 'fc_3layersnpd':
      return mnist_models.SNPDFC3        
    else:
      raise ValueError(f'Model name not recognized ({decoder_name})')
  elif 'cifar' in dataset:
    if decoder_name == 'fc_3layersnpd':
      return cifar_models.SNPDFC3
    else:
      raise ValueError(f'Model name not recognized ({decoder_name})')
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')


def get_train_args(dataset, opt):
  trainargs = {}
  if dataset == 'mnist':
    trainargs['num_decoder_feats'] = 64
    trainargs['num_classes'] = 10
    trainargs['train_batch_size'] = 128
    trainargs['test_batch_size'] = 1000
    trainargs['log_interval'] = 500
    trainargs['weight_decay'] = 1e-5
    trainargs['save_interval'] = 10
    if opt == 'sgd':
      trainargs['nb_epoch'] = 100
      trainargs['e_lr'] = 0.5
      trainargs['edc_lr'] = 0.1
      trainargs['da_lr'] = 0.5
      trainargs['schedule_milestones'] = [50, 80]
      trainargs['weight_decay'] = 1e-4
      trainargs['scheduler_gamma'] = 0.1
    elif opt == 'adam':
      trainargs['nb_epoch'] = 200
      trainargs['e_lr'] = 1e-4
      trainargs['edc_lr'] = 1e-4
      trainargs['da_lr'] = 1e-3
      trainargs['schedule_milestones'] = [100]
      trainargs['scheduler_gamma'] = 0.1
    else:
      raise ValueError(f'optimizer not recognized {opt}. Only sgd or adam.')
  elif 'cifar' in dataset:
    trainargs['num_decoder_feats'] = 512
    if dataset == 'cifar10':
      trainargs['num_classes'] = 10
    else:
      trainargs['num_classes'] = 100
    trainargs['train_batch_size'] = 128
    trainargs['test_batch_size'] = 128
    trainargs['log_interval'] = 200
    trainargs['nb_epoch'] = 300
    trainargs['e_lr'] = 0.5
    trainargs['edc_lr'] = 0.1
    trainargs['da_lr'] = 0.1
    trainargs['weight_decay'] = 1e-4
    trainargs['schedule_milestones'] = [150, 250]
    trainargs['scheduler_gamma'] = 0.1
    trainargs['save_interval'] = 10
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')
  return trainargs


def safewrite_json(filepath, new_rows, update=False, crit=None):
  """
  safely writes into a pandas data frame stored in json format.
  :param filepath: path to json file
  :param new_rows: contains the rows to be inserted or the values to be updates if update=True
  :param update: if True updates the rows defined by crit with values from new_rows
  :param crit: criterion to update the rows
  :return:
  """
  parent = Path(filepath).parent
  if not os.path.exists(parent):
    os.makedirs(parent)
  try:
    with FileLock(filepath + '.lock'):
      if not os.path.exists(filepath):
        df = pd.DataFrame(new_rows)
        df.to_json(filepath)
      else:
        df = pd.read_json(filepath)
        if update:
          df.loc[(df[list(crit)] == pd.Series(crit)).all(axis=1), list(new_rows)] = [*new_rows.values()]
        else:
          df = df.append(new_rows, ignore_index=True)
        df.to_json(filepath)
  except Exception as e:
    print('save failed')
    raise e


def get_mnist_train_loader(batch_size, data_path, shuffle=True):
  ts = [
    transforms.ToTensor()
  ]
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=True, download=True,
                   transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle)
  loader.name = "mnist_train"
  return loader


def get_mnist_test_loader(batch_size, data_path, shuffle=False):
  loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=shuffle)
  loader.name = "mnist_test"
  return loader


def get_cifar10_train_loader(batch_size, data_path, shuffle=True, norm=False):
  ts = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_path, train=True, download=True,
                     transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=8)
  loader.name = "cifar10_train"
  return loader


def get_cifar10_test_loader(batch_size, data_path, shuffle=False, norm=False):
  ts = [transforms.ToTensor()]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  test_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_path, train=False, download=True,
                     transform=test_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=2)
  loader.name = "cifar10_test"
  return loader


def get_cifar100_train_loader(batch_size, data_path, shuffle=True, norm=False):
  ts = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_path, train=True, download=True,
                     transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=8)
  loader.name = "cifar10_train"
  return loader


def get_cifar100_test_loader(batch_size, data_path, shuffle=False, norm=False):
  ts = [transforms.ToTensor()]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  test_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_path, train=False, download=True,
                     transform=test_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=2)
  loader.name = "cifar10_test"
  return loader


def get_unshared_pars(m1, m2):
  m1pars = []
  m1par_names = []
  for n, m in m1.named_parameters():
    m1pars.append(m)
    m1par_names.append(n)

  m2pars = []
  m2par_names = []
  for n, m in m2.named_parameters():
    m2pars.append(m)
    m2par_names.append(n)

  return list(set(m1pars) - set(m2pars)), list(set(m1par_names) - set(m2par_names))


def get_decoder_pars(model):
  pars = []
  par_names = []
  for n, m in model.named_parameters():
    if 'encoder' in n:
      continue
    pars.append(m)
    par_names.append(n)

  return pars, par_names

  
class KLPGDAttack: 
  def __init__(self, model, eps_iter=0.007, eps=0.031, nb_iter=5, clip_min=0., clip_max=1.0, distance='l_inf'):  
    self.model=model
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter=nb_iter
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.distance=distance    
     
  def perturb(self, x_natural, target=None):
      # define KL-loss
      criterion_kl = nn.KLDivLoss(reduction='sum')
      batch_size = len(x_natural)
      # generate adversarial example
      x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
      if self.distance == 'l_inf':
          for _ in range(self.nb_iter):
              x_adv.requires_grad_()
              with torch.enable_grad():
                  loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                        F.softmax(self.model(x_natural), dim=1))
              grad = torch.autograd.grad(loss_kl, [x_adv])[0]              
              x_adv = x_adv.detach() + self.eps_iter * torch.sign(grad.detach())
              x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
              x_adv = torch.clamp(x_adv, 0.0, 1.0)
      elif self.distance == 'l_2':
          delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
          delta = Variable(delta.data, requires_grad=True)

          # Setup optimizers
          optimizer_delta = optim.SGD([delta], lr=self.eps / self.nb_iter * 2)

          for _ in range(self.nb_iter):
              adv = x_natural + delta

              # optimize
              optimizer_delta.zero_grad()
              with torch.enable_grad():
                  loss = (-1) * criterion_kl(F.log_softmax(self.model(adv), dim=1),
                                            F.softmax(self.model(x_natural), dim=1))
              loss.backward()
              # renorming gradient
              grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
              delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
              # avoid nan or inf if gradient is 0
              if (grad_norms == 0).any():
                  delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
              optimizer_delta.step()

              # projection
              delta.data.add_(x_natural)
              delta.data.clamp_(0, 1).sub_(x_natural)
              delta.data.renorm_(p=2, dim=0, maxnorm=self.eps)
          x_adv = Variable(x_natural + delta, requires_grad=False)
      else:
          x_adv = torch.clamp(x_adv, 0.0, 1.0)
      return x_adv
class AA(AutoAttack): 
  def __init__(self, model, attacks_to_run=[], norm='Linf', eps=0.3, n_iter=20, version='standard', verbose=False):
    super(AA, self).__init__(model, attacks_to_run=attacks_to_run, 
    norm=norm, eps=eps, n_iter=n_iter, version=version, verbose=verbose)

  def perturb(self, x_orig, y_orig): 
    return self.run_standard_evaluation_individual(x_orig, y_orig, bs=x_orig.shape[0])[self.attacks_to_run[0]]

