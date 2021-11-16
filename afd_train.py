"""
Train with adversarial feature desensitization
"""
from __future__ import print_function

import os
import argparse
import datetime
import copy
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from torch.utils.tensorboard import SummaryWriter
from utils import get_attack, get_model, get_decoder_class, \
  get_decoder_pars, get_data_loaders, get_train_args, prepare_dirs
import losses


def log_outputs(data_loader, model, it, device, writer, adversary): 
  model.eval()
  test_clnloss = 0
  clncorrect = 0
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

  test_advloss /= len(data_loader.dataset)
  adv_acc = 100. * advcorrect / len(data_loader.dataset)
  print('Test set: avg adv loss: {:.4f},'
        ' adv acc: {}/{} ({:.0f}%)\n'.format(
    test_advloss, advcorrect, len(data_loader.dataset),
    adv_acc))
  writer.add_scalar('Loss/C_test_adv', test_advloss, it)
  writer.add_scalar('Test Accuracy/adv', adv_acc, it)


def get_optimizers(args, trainargs, model_pars):
  if args.optimizer == 'adam':
    if args.gan_loss_type == "wgan_gp":
      beta1 = 0.0
      beta2 = 0.9
    else:
      beta1 = 0.5
      beta2 = 0.999

    optDa = optim.Adam(model_pars['Da_pars'], lr=trainargs['da_lr'], betas=(beta1, beta2))
    optE = optim.Adam(model_pars['E_pars'], lr=trainargs['e_lr'], betas=(beta1, beta2))
    optEDc = optim.Adam(model_pars['EDc_pars'], lr=trainargs['edc_lr'], betas=(beta1, beta2))
    Dalr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optDa, trainargs['schedule_milestones'],
                                                          gamma=trainargs['scheduler_gamma'],
                                                          last_epoch=-1)
    Elr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optE, trainargs['schedule_milestones'],
                                                         gamma=trainargs['scheduler_gamma'],
                                                         last_epoch=-1)
    EDclr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optEDc, trainargs['schedule_milestones'],
                                                           gamma=trainargs['scheduler_gamma'],
                                                           last_epoch=-1)

  else:
    optDa = optim.SGD(model_pars['Da_pars'], lr=trainargs['da_lr'], momentum=0.9)
    optE = optim.SGD(model_pars['E_pars'], lr=trainargs['e_lr'], momentum=0.9)
    optEDc = optim.SGD(model_pars['EDc_pars'], lr=trainargs['edc_lr'], momentum=0.9, 
    weight_decay=trainargs['weight_decay']
    )
    Dalr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optDa, trainargs['schedule_milestones'],
                                                          gamma=trainargs['scheduler_gamma'],
                                                          last_epoch=-1)
    Elr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optE, trainargs['schedule_milestones'],
                                                         gamma=trainargs['scheduler_gamma'],
                                                         last_epoch=-1)
    EDclr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optEDc, trainargs['schedule_milestones'],
                                                           gamma=trainargs['scheduler_gamma'],
                                                           last_epoch=-1)
  return optDa, optE, optEDc, Dalr_scheduler, Elr_scheduler, EDclr_scheduler


def train(args):
  TRAINED_MODEL_PATH, DATA_PATH = prepare_dirs(args.save_path, args.dataset, args.exp_name)
  writer = SummaryWriter(TRAINED_MODEL_PATH)

  torch.manual_seed(args.seed)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model_filename = f"{args.dataset}_model"

  trainargs = get_train_args(args.dataset, args.optimizer)
  train_loader, test_loader = get_data_loaders(args.dataset,
                                               trainargs['train_batch_size'], trainargs['test_batch_size'],
                                               DATA_PATH)

  E, Dc = get_model(args.dataset, args.enc_model, num_decoder_features=trainargs['num_decoder_feats'])
  EDc = nn.Sequential(E, Dc)
  Da_class = get_decoder_class(args.dataset, args.advdec_model)

  EDa = Da_class(E, num_features=trainargs['num_decoder_feats'], num_classes=trainargs['num_classes'])
  Da_pars, Da_par_names = get_decoder_pars(EDa)

  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    EDc = nn.DataParallel(EDc)
    EDa = nn.DataParallel(EDa)
  EDc.to(device)
  EDa.to(device)

  optDa, optE, optEDc, Dalr_scheduler, Elr_scheduler, EDclr_scheduler = \
    get_optimizers(args, 
    trainargs, 
    {'Da_pars': Da_pars, 'E_pars': E.parameters(), 'EDc_pars':EDc.parameters()}
    )
  
  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch']
      it = checkpoint['iteration']
      E.load_state_dict(checkpoint['E_state_dict'])
      Dc.load_state_dict(checkpoint['Dc_state_dict'])
      EDa.load_state_dict(checkpoint['EDa_state_dict'])
      optEDc.load_state_dict(checkpoint['optEDc'])
      optE.load_state_dict(checkpoint['optE'])
      optDa.load_state_dict(checkpoint['optDa'])

      # run scheduler steps
      for _ in range(start_epoch):
        Dalr_scheduler.step()
        Elr_scheduler.step()
        EDclr_scheduler.step()

      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
  else:
    start_epoch = 0
    it = 0

  adversary = get_attack(EDc, args.attack_name, args.dataset)
  E_criterion = losses.GenLoss(args.gan_loss_type)
  Da_criterion = losses.DisLoss(args.gan_loss_type)

  # Start the training loop
  for epoch in range(start_epoch, trainargs['nb_epoch']):
    adv_train_correct = []
    EDc.train()
    EDa.train()

    for batch_idx, (data, target) in enumerate(train_loader):
      rand_it_num = torch.randint(0, 100, (1,))
      data, target = data.to(device), target.to(device)

      with ctx_noparamgrad_and_eval(EDc):
        adv_data = adversary.perturb(data, target).detach()

      # Train EDc for classification
      optEDc.zero_grad()
      output = EDc(data)
      errC = F.cross_entropy(
        output, target, reduction='mean')
      errC.backward()
      optEDc.step()

      if (epoch % trainargs['save_interval'] == 0) and (batch_idx == 0):
        for n, p in EDc.named_parameters():
          writer.add_histogram(f'EDc_grads/{n}', p.grad.data, it)
          writer.add_histogram(f'EDc_weights/{n}', p.data, it)

      # Train Da for adversary decoding
      optDa.zero_grad()
      output_clean = EDa(data, target)
      output_adv = EDa(adv_data, target)
      D_x = output_clean.mean().item()
      D_x_adv = output_adv.mean().item()
      errD = Da_criterion(output_adv.view(-1), output_clean.view(-1))
      if args.gan_loss_type == "wgan_gp":
        gp = losses.gradient_penalty(EDa, E(data).detach(), E(adv_data).detach(), device=device)
        errD += args.dis_lambda * gp
      errD.backward()
      optDa.step()

      if (epoch % trainargs['save_interval'] == 0) and (batch_idx == 0):
        for n, p in zip(Da_par_names, Da_pars):
          writer.add_histogram(f'Da_grads/{n}', p.grad.data, it)
          writer.add_histogram(f'Da_weights/{n}', p.data, it)

      # Train E for (against) adversary decoding
      optE.zero_grad()
      output_adv = EDa(adv_data, target)
      with torch.no_grad():
        EDc_output = EDc(adv_data)
      EDc_train_pred = EDc_output.max(1, keepdim=True)[1]
      adv_train_correct.append(
        100. * EDc_train_pred.eq(target.view_as(EDc_train_pred)).sum().item() / float(len(EDc_train_pred)))
      errE = E_criterion(output_adv.view(-1), None)
      if rand_it_num % args.dis_iters == 0:
        errE.backward()
        optE.step()

      if (epoch % trainargs['save_interval'] == 0) and (batch_idx == 0):
        for n, p in E.named_parameters():
          writer.add_histogram(f'E_grads/{n}', p.grad.data, it)
          writer.add_histogram(f'E_weights/{n}', p.data, it)

      # Log outputs
      if batch_idx % trainargs['log_interval'] == 1:
        print(
          f'{datetime.datetime.now()} Train Epoch: {epoch} '
          f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
          f'({100. * batch_idx / len(train_loader):.0f}%)]')
        print(f'Loss_C: {errC.item():.4f}\tLoss_Da: {errD.item():.4f}'
              f'\tLoss_E: {errE.item():.4f}\tDa(x): {D_x_adv:.4f} / {D_x:.4f}')
        writer.add_scalar('Loss/C', errC.item(), it)
        writer.add_scalar('Loss/D', errD.item(), it)
        writer.add_scalar('Loss/E', errE.item(), it)
        writer.add_scalar('Da/x', D_x, it)
        writer.add_scalar('Da/x_adv', D_x_adv, it)
      it += 1
    writer.add_scalar('Train Accuracy/adv', np.mean(adv_train_correct), it)

    # Evaluate model
    log_outputs(test_loader, EDc, it, device, writer, adversary)

    # Save checkpoint
    if (epoch % trainargs['save_interval'] == 0) or (epoch == (trainargs['nb_epoch'] - 1)):
      checkpoint = {'E_state_dict': E.state_dict(),
                    'Dc_state_dict': Dc.state_dict(),
                    'EDa_state_dict': EDa.state_dict(),
                    'epoch': epoch + 1,
                    'iteration': it,
                    'enc_model': args.enc_model,
                    'optEDc': optEDc.state_dict(),
                    'optE': optE.state_dict(),
                    'optDa': optDa.state_dict()
                    }
      torch.save(checkpoint,
                 os.path.join(TRAINED_MODEL_PATH, model_filename + f'_{epoch}.pt'))
      torch.save(checkpoint,
                 os.path.join(TRAINED_MODEL_PATH, model_filename + f'_last.pt'))

    # scheduler update
    if args.optimizer == 'sgd':
      Dalr_scheduler.step()
      Elr_scheduler.step()
      EDclr_scheduler.step()
      writer.add_scalar('LR', Elr_scheduler.get_last_lr()[0], it)
  writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='AFD training')
  parser.add_argument('--save_path', default='./chkpts', type=str, help='path to where to save checkpoints')
  parser.add_argument('--optimizer', default='sgd', type=str, help='sgd | adam')
  parser.add_argument('--dataset', default='cifar10', type=str,
                      help='mnist | cifar10 | cifar100')
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('--enc_model', default='resnet18norm', type=str)
  parser.add_argument('--advdec_model', default='fc_3layersnpd', type=str)
  parser.add_argument('--attack_name', default='aa_apgdce', type=str)
  parser.add_argument('--gan_loss_type', type=str, default='wgan_gp',
                      help='loss function name. dcgan (default) | hinge | wgan_gp .')
  parser.add_argument('--dis_iters', type=int, default=1)
  parser.add_argument('--dis_lambda', type=float, default=0.)
  parser.add_argument('--exp_name', default='resnet18norm_wgan_sgd', type=str, help='experiment name')
  args = parser.parse_args()
  nt = namedtuple('nt', [*args.__dict__.keys()])
  train(nt(*args.__dict__.values()))
