from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from models.sem_gcn_equiv import SemGCNEquiv

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for strictly equivariant GCN')
    
    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D keypoints to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 5)')
    
    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='learning rate decay rate')
    parser.add_argument('--no_max', dest='max_norm', action='store_false',
                        help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor')

    args = parser.parse_args()
    return args

def main(args):
    print('==> Using settings {}'.format(args))
    
    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')
        
    print('==> Preparing data...')
    dataset = read_3d_data(dataset)
    
    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)
    
    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))
        
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")
    
    # Create model
    print("==> Creating model...")
    
    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCNEquiv(adj, args.hid_dim,
                           num_layers=args.num_layers,
                           p_dropout=p_dropout).to(device)
    
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))
    
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)
    
    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = args.resume if args.resume else args.evaluate
        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint '{}' (epoch {})".format(ckpt_path, start_epoch))
            
            if args.evaluate:
                errors_p1, errors_p2 = evaluate(test_generator, model_pos, device)
                print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
                print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
                return
            
        else:
            print("==> No checkpoint found at '{}'".format(ckpt_path))
            return
    
    # Create data generators
    print('==> Loading data...')
    train_generator = PoseGenerator(dataset, keypoints, args.batch_size, args.num_workers, True, args.downsample, subset=subjects_train)
    test_generator = PoseGenerator(dataset, keypoints, args.batch_size, args.num_workers, False, args.downsample, subset=subjects_test)
    
    if args.evaluate:
        errors_p1, errors_p2 = evaluate(test_generator, model_pos, device)
        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        return
    
    # Training initialization
    print('==> Training...')
    start_epoch = 0
    error_best = None
    glob_step = 0
    lr_now = args.lr
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint, exist_ok=True)
    
    # Create logger
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
    logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))
        
        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_generator, model_pos, criterion, optimizer, device,
                                            args.lr, lr_now, glob_step, args.lr_decay, args.lr_gamma, args.max_norm)
        
        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(test_generator, model_pos, device)
        
        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])
        
        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                      'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, checkpoint=args.checkpoint,
                     snapshot=args.snapshot)
        
        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                      'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, checkpoint=args.checkpoint,
                     snapshot=epoch + 1)

def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()
    
    bar = Bar('Eval ', max=len(data_loader))
    all_dist_p1 = []
    all_dist_p2 = []
    
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        
        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d).cpu()
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)
        
        # Calculate MPJPE
        dist_p1 = mpjpe(outputs_3d, targets_3d).item() * 1000.0
        dist_p2 = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0
        
        all_dist_p1.append(dist_p1)
        all_dist_p2.append(dist_p2)
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'.format(
            batch=i + 1,
            size=len(data_loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            e1=np.mean(all_dist_p1),
            e2=np.mean(all_dist_p2)
        )
        bar.next()
    
    bar.finish()
    return np.mean(all_dist_p1), np.mean(all_dist_p2)

def train(data_loader, model_pos, criterion, optimizer, device,
          lr_init, lr_now, step, decay, gamma, max_norm):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    
    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()
    
    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        
        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)
            
        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d)
        
        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()
        
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
            batch=i + 1,
            size=len(data_loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=epoch_loss_3d_pos.avg
        )
        bar.next()
    
    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step

if __name__ == '__main__':
    args = parse_args()
    main(args)
