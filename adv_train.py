from __future__ import print_function

import os
import argparse
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_model, get_data_loaders, get_attack, prepare_dirs
from advertorch.context import ctx_noparamgrad_and_eval
from torch.utils.tensorboard import SummaryWriter


def get_train_pars(mode, dataset, enc_model): 
  if mode == "cln":
    flag_advtrain = False
    if dataset == 'mnist': 
        nb_epoch = 20
        lr_steps = [40, 80]
    else: 
        nb_epoch = 300
        lr_steps = [150, 250]
    model_filename = f"{dataset}_{enc_model}_clntrained.pt"
  elif mode == "adv":
    flag_advtrain = True
    if dataset == 'mnist': 
        nb_epoch = 90
        lr_steps = [40, 80]
    else: 
        nb_epoch = 300
        lr_steps = [150, 250]
    model_filename = f"{dataset}_{enc_model}_advtrained.pt"
  else:
    raise ValueError
  return flag_advtrain, nb_epoch, lr_steps, model_filename


def log_outputs(data_loader, model, flag_advtrain, it, device, writer, adversary): 
      model.eval()
      test_clnloss = 0
      clncorrect = 0

      if flag_advtrain:
        test_advloss = 0
        advcorrect = 0

      for clndata, target in data_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
          output = model(clndata)
        test_clnloss += F.cross_entropy(
          output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        if flag_advtrain:
          advdata = adversary.perturb(clndata, target)
          with torch.no_grad():
            output = model(advdata)
          test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
          pred = output.max(1, keepdim=True)[1]
          advcorrect += pred.eq(target.view_as(pred)).sum().item()

      test_clnloss /= len(data_loader.dataset)
      cln_acc = 100. * clncorrect / len(data_loader.dataset)
      print('\nTest set: avg cln loss: {:.4f},'
            ' cln acc: {}/{} ({:.0f}%)\n'.format(
          test_clnloss, clncorrect, len(data_loader.dataset),
          cln_acc))
      writer.add_scalar('Loss/C_test_cln', test_clnloss, it)
      writer.add_scalar('Test Accuracy/cln', cln_acc, it)

      if flag_advtrain:
        test_advloss /= len(data_loader.dataset)
        adv_acc = 100. * advcorrect / len(data_loader.dataset)
        print('Test set: avg adv loss: {:.4f},'
              ' adv acc: {}/{} ({:.0f}%)\n'.format(
            test_advloss, advcorrect, len(data_loader.dataset),
            adv_acc))
        writer.add_scalar('Loss/C_test_adv', test_advloss, it)
        writer.add_scalar('Test Accuracy/adv', adv_acc, it)


def train(args):
  TRAINED_MODEL_PATH, DATA_PATH = prepare_dirs(args.save_path, args.dataset, args.exp_name)
  writer = SummaryWriter(TRAINED_MODEL_PATH)

  torch.manual_seed(args.seed)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  flag_advtrain, nb_epoch, lr_steps, model_filename = get_train_pars(args.mode, args.dataset, args.enc_model)
  train_loader, test_loader = get_data_loaders(args.dataset,
                                               args.train_batch_size, args.test_batch_size,
                                               DATA_PATH)

  E, Dc = get_model(args.dataset, args.enc_model)
  model = nn.Sequential(E, Dc)

  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

  model.to(device)
  opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, lr_steps, gamma=0.1, last_epoch=-1)

  if flag_advtrain:
    adversary = get_attack(model, args.attack_name, args.dataset)
  else: 
    adversary = None

  it = 0
  for epoch in range(nb_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      if flag_advtrain:
        with ctx_noparamgrad_and_eval(model):
          data = adversary.perturb(data, target)

      opt.zero_grad()
      output = model(data)
      loss = F.cross_entropy(
        output, target, reduction='mean')
      loss.backward()
      opt.step()
      if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx *
                 len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
        writer.add_scalar('Loss/C', loss.item(), it)

    it += 1

    # Model evaluation
    log_outputs(test_loader, model, flag_advtrain, it, device, writer, adversary)

    # Save model checkpoint
    if (epoch % 10 == 0) or (epoch == (nb_epoch - 1)):
      torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename + f'_{epoch}.pt'))
    lr_scheduler.step()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train NT and AT')
  parser.add_argument('--save_path', default='./chkpts', type=str, help='path to where to save checkpoints')
  parser.add_argument('--dataset', default='cifar10', type=str, help='mnist | cifar10 | cifar100')
  parser.add_argument('--attack_name', default='aa_apgdce', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--mode', default="adv", help="cln | adv")
  parser.add_argument('--train_batch_size', default=128, type=int)
  parser.add_argument('--test_batch_size', default=128, type=int)
  parser.add_argument('--log_interval', default=200, type=int)
  parser.add_argument('--enc_model', default='resnet18norm', type=str)
  parser.add_argument('--exp_name', default='adv_train_resnet18norm', type=str, help='experiment name')
  args = parser.parse_args()
  nt = namedtuple('nt', [*args.__dict__.keys()])
  train(nt(*args.__dict__.values()))            
