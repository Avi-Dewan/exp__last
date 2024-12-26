# -*- coding: utf-8 -*-
import os
import gc
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.losses import SimclrCriterion
from optimisers import get_optimiser


def load_checkpoint(checkpoint_path, encoder, mlp, optimiser):
    ''' Load checkpoint from a given file path '''
    if not os.path.exists(checkpoint_path):
        logging.info(f'Checkpoint file {checkpoint_path} not found. Starting from scratch.')
        return 0  # Start from epoch 0

    logging.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder'])
    mlp.load_state_dict(checkpoint['mlp'])
    optimiser.load_state_dict(checkpoint['optimiser'])

    return checkpoint['epoch'] + 1




def pretrain(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SimCLR

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SimclrCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    start_epoch = 0

    ''' Load checkpoint if available '''
    if args.checkpoint_path:
        start_epoch = load_checkpoint(args.checkpoint_path, encoder, mlp, optimiser)

    '''Create checkpoint dir to save checkpints'''
    args.checkpoint_dir = './checkpoint'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ''' Pretrain loop '''
    for epoch in range(start_epoch, args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        ''' epoch loop '''
        for i, (inputs, _) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            # retrieve the 2 views
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            # Get the encoder representation
            _, h_i = encoder(x_i)
            _, h_j = encoder(x_j)

            # Get the nonlinear transformation of the representation
            z_i = mlp(h_i)
            z_j = mlp(h_j)

            # Calculate NT_Xent loss
            loss = criterion(z_i, z_j)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        ''' Save checkpoint at intervals '''
        if (epoch + 1) % args.save_interval == 0:
            epoch_checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch + 1}.pth')
            state = {
                'encoder': encoder.state_dict(),
                'mlp': mlp.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, epoch_checkpoint_path)

        if(epoch+1 == args.n_epochs):
            last_model_path = os.path.join(args.checkpoint_dir, f'resnet_epoch{args.n_epochs}.pth')
            torch.save(encoder.state_dict(), last_model_path)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

            # Save only the encoder
            best_model_path = os.path.join(args.checkpoint_dir, f'resnet.pth')
            torch.save(encoder.state_dict(), best_model_path)

            print(f'Got the best model at epoch: {best_epoch}')

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_pretrain_loss = None  # reset loss


    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory



def evaluate(encoder, mlp, dataloaders, mode, epoch, args):
    ''' Evaluate script - SimCLR

        evaluate the encoder and classification head with Cross Entropy loss.
    '''

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc
    epoch_valid_acc_top5 = None

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()
    mlp.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    ''' epoch loop '''
    for i, (inputs, target) in enumerate(eval_dataloader):

        # Do not compute gradient for encoder and classification head
        encoder.zero_grad()
        mlp.zero_grad()

        inputs = inputs.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # Forward pass

        _, h = encoder(inputs)

        output = mlp(h)

        loss = criterion(output, target)

        torch.cuda.synchronize()

        sample_count += inputs.size(0)

        run_loss += loss.item()

        predicted = output.argmax(-1)

        acc = (predicted == target).sum().item() / target.size(0)

        run_top1 += acc

        _, output_topk = output.topk(5, 1, True, True)

        acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / target.size(0)  # num corrects

        run_top5 += acc_top5

    epoch_valid_loss = run_loss / len(dataloaders[mode])  # sample_count

    epoch_valid_acc = run_top1 / len(dataloaders[mode])

    epoch_valid_acc_top5 = run_top5 / len(dataloaders[mode])

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[{}] loss: {:.4f},\t acc: {:.4f},\t acc_top5: {:.4f} \n'.format(
            mode, epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5))

        if mode != 'test':
            args.writer.add_scalars('finetune_epoch_loss', {mode: epoch_valid_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {mode: epoch_valid_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top5', {
                                    'train': epoch_valid_acc_top5}, epoch+1)

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

    return epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5
