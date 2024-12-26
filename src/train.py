# -*- coding: utf-8 -*-
import os
import gc
import logging
import sys
import numpy as np
from tqdm import tqdm


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the system path
sys.path.append(parent_dir)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.losses import SimclrCriterion
from optimisers import get_optimiser


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.util import cluster_acc
from dataloader.cifarloader import CIFAR10Loader, CIFAR100Loader

def plot_features_And_calculate_metric(model, test_loader, save_path, epoch, device, args):
    torch.manual_seed(1)
    model = model.to(device)
    model.eval()
    targets = np.array([])
    outputs = np.zeros((len(test_loader.dataset), 512 )) 
    
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        _, output = model(x)
       
        outputs[idx, :] = output.cpu().detach().numpy()
        targets = np.append(targets, label.cpu().numpy())

    # print("Unique labels:", np.unique(targets))

    pca = PCA(n_components=20) # PCA for dimensionality reduction PCA: 512 -> 20
    pca_features = pca.fit_transform(outputs) # fit the PCA model and transform the features
    kmeans = KMeans(n_clusters=args.n_unlabeled_classes, n_init=20)  # KMeans clustering
    y_pred = kmeans.fit_predict(pca_features)

    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)

    # Normalize targets for categorical mapping
    targets_normalized = (targets - targets.min()).astype(int)  # Map to range 0-19

    # Create t-SNE visualization
    X_embedded = TSNE(n_components=2).fit_transform(outputs)

    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets_normalized, cmap='tab20')
    plt.colorbar(scatter)  # Add color bar to verify mapping
    plt.title(f"t-SNE Visualization of Features on {args.dataset_name} - Epoch {epoch}")
    plt.savefig(f"{save_path}/{args.dataset_name}_epoch{epoch}.png")

    return acc, nmi, ari



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

    return checkpoint['epoch'] 


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

    '''loading unlabeled datas to check cluster quality and tsne plot'''
    if args.dataset == 'cifar10':
        dloader_unlabeled_test = CIFAR10Loader(
            root=args.dataset_root, 
            batch_size=128, 
            split='test', 
            aug=None, 
            shuffle=False, 
            target_list = range(5, 10))
        
        args.n_unlabeled_classes = 5

    elif args.dataset == 'cifar100':
        dloader_unlabeled_test = CIFAR100Loader(
            root=args.dataset_root, 
            batch_size=128, 
            split='test', 
            aug=None, 
            shuffle=False, 
            target_list = range(25, 50))
        
        args.n_unlabeled_classes = 25
    

    start_epoch = 0

    ''' Load checkpoint if available '''
    if args.checkpoint_path:
        start_epoch = load_checkpoint(args.checkpoint_path, encoder, mlp, optimiser)

    '''Create checkpoint dir to save checkpints'''
    args.checkpoint_dir = './checkpoints'
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
                'epoch': epoch+1,
            }
            torch.save(state, epoch_checkpoint_path)
            print(f"Checkpoint saved at epoch-{epoch+1}")

            acc, nmi, ari = plot_features_And_calculate_metric(encoder, dloader_unlabeled_test, 
                           args.checkpoint_dir, epoch+1, args.device, args)
            
            print("-------------------------------------")
            print(f'Epoch-{epoch+1}: ACC = {acc} , NMI = {nmi}, ARI = {ari} ')
            print("-------------------------------------")

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
