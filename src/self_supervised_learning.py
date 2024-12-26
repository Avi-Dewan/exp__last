#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from train import pretrain
from datasets import get_dataloaders
from util import experiment_config, print_network, init_weights
import model.network as models
from model.resnet import ResNet, BasicBlock, Identity

warnings.filterwarnings("ignore")

default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch SimCLR', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--finetune_learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='lars',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--finetune_optimiser', default='sgd',
                    help='Finetune Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--temperature', type=float, default=0.5, help='NT_Xent Temperature Factor')
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)


parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
parser.set_defaults(finetune=False)
parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')

parser.add_argument('--save_interval', type=int, default=50,
                        help='Interval to save the model and plot tsne(default: 50)')
parser.add_argument('--plot_interval', type=int, default=50,
                        help='Interval to save the model and plot tsne(default: 50)')
parser.add_argument('--checkpoint_path', type = str, default='',
                    help='Path to Load Pre-trained Model From.')

parser.set_defaults(supervised=False)


def setup():
    """ 
    Sets up for training.

    """
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 420
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device


def main():
    """ Main """

    # Arguments
    args = parser.parse_args()

    # Setup  Training
    device = setup()
    args.device = device

    # Get Dataloaders for Dataset of choice
    dataloaders, args = get_dataloaders(args)

    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    if args.model == 'resnet18':
        base_encoder = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=args.n_classes) # Resnet 18
        proj_head = models.projection_MLP(args)
    
    base_encoder.linear = Identity()
    
    # If multiple GPUs, use DataParallel
    if torch.cuda.device_count() > 1:
        base_encoder = nn.DataParallel(base_encoder)
        proj_head = nn.DataParallel(proj_head)

    print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

    base_encoder.to(device)
    proj_head.to(device)

    args.print_progress = True

    # Print Network Structure and Params
    if args.print_progress:
        print_network(base_encoder, args)  # prints out the network architecture etc
        logging.info('\npretrain/train: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    # launch model training or inference
    
    pretrain(base_encoder, proj_head, dataloaders, args)
        

if __name__ == '__main__':
    main()
