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
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from models.grit_model import GRITModel
from models.gine_gcn import GINEGCN
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from random import randrange

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for GRIT-based Human Pose Estimation')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs')
    parser.add_argument('--perm_type', default='complex', type=str, choices=['trivial', 'identity', 'complex', 'complete','disjoint'],
                        help='permutation group type: trivial (body symmetry), complex (arm/leg symmetry), complete (all symmetry), disjoint (arm/leg joints alone)')
    parser.add_argument('--maxnorm', default=20, type=int, help='max_norm_main')
    parser.add_argument('--sparse', action='store_true', 
                    help='Enable sparse model')
    parser.add_argument('--soft', action='store_true', 
                    help='Soft Sparsity')

    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='number of layers in GRIT')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='number of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--subset', default=1.0, type=float, help='use a subset of the training data')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

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
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.perm_type == 'complex':
        perms = [
            Permutation([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),  
            Permutation([0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15]),
            Permutation([0,1,2,3,4,5,6,7,8,9,13,14,15,10,11,12])
        ]
    elif args.perm_type == 'disjoint':
        perms = [
            Permutation([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),  
            Permutation([0,4,2,3,1,5,6,7,8,9,10,11,12,13,14,15]),
            Permutation([0,1,5,3,4,2,6,7,8,9,10,11,12,13,14,15]),
            Permutation([0,1,2,6,4,5,3,7,8,9,10,11,12,13,14,15]),
            Permutation([0,1,2,3,4,5,6,7,8,9,13,11,12,10,14,15]),
            Permutation([0,1,2,3,4,5,6,7,8,9,10,14,12,13,11,15]),
            Permutation([0,1,2,3,4,5,6,7,8,9,10,11,15,13,14,12])
        ]
    elif args.perm_type == 'simple':
        perms = [
            Permutation([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),  
            Permutation([0,4,5,6,1,2,3,7,8,9,13,14,15,10,11,12])
        ]
    elif args.perm_type == 'identity':
        perms = [
            Permutation([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        ]
    elif args.perm_type == 'complete':
        S16 = SymmetricGroup(16)
        perms = S16.generators

    # Create model
    print("==> Creating GRIT model...")
    model_pos = GRITModel(
        n_nodes=dataset.skeleton().num_joints(),  
        hid_dim=args.hid_dim,
        perms = perms,
        p_dropout=args.dropout,
        sparse = args.sparse,
        maxnorm = args.maxnorm,
        soft = args.soft
    ).to(device)

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
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])
        # Log additional information before appending epoch data
        with open(os.path.join(ckpt_dir_path, 'log.txt'), 'a') as log_file:
            log_file.write(f"args.perm: {args.perm_type}\n")
            log_file.write(f"args.sparse: {args.sparse}\n")
            log_file.write(f"args.soft: {args.soft}\n")
            log_file.write("GAT\n")

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i] = evaluate(valid_loader, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {} mm'.format(np.mean(errors_p1)))
        print('Protocol #2 (P-MPJPE) action-wise average: {} mm'.format(np.mean(errors_p2)))
        exit(0)

    poses_train, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)
    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train, args.subset),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

    poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train
        epoch_loss = train(train_loader, model_pos, criterion, optimizer, device, args.max_norm)


        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update learning rate
        lr_now = lr_decay(optimizer, epoch + 1, lr_now, args.lr_decay, args.lr_gamma)

        # Save checkpoint
        if error_best is None or error_eval_p1 < error_best:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                      'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                      'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])
    logger.close()

def train(train_loader, model_pos, criterion, optimizer, device, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(train_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=epoch_loss_3d_pos.avg
        )
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg

def evaluate(valid_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(valid_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        with torch.no_grad():
            targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
            outputs_3d = model_pos(inputs_2d)

            outputs_3d = outputs_3d.cpu()
            targets_3d = targets_3d.cpu()
            
            loss_3d_pos = mpjpe(outputs_3d, targets_3d).item() * 1000.0
            loss_3d_pos_procrustes = p_mpjpe(outputs_3d, targets_3d).item() * 1000.0

            epoch_loss_3d_pos.update(loss_3d_pos, num_poses)
            epoch_loss_3d_pos_procrustes.update(loss_3d_pos_procrustes, num_poses)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | MPJPE: {e1:.3f} | P-MPJPE: {e2:.3f}'.format(
            batch=i + 1,
            size=len(valid_loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            e1=epoch_loss_3d_pos.avg,
            e2=epoch_loss_3d_pos_procrustes.avg
        )
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg

if __name__ == '__main__':
    args = parse_args()
    main(args)
