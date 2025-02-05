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
from models.gine_gcn import GINEGCN
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from random import randrange

def parse_args():
    parser = argparse.ArgumentParser(description='Human Pose Estimation')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST', help='actions to train/test on')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    # Model arguments (Not used, as we are simplifying the code)
    parser.add_argument('--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')

    args = parser.parse_args()
    return args

def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = os.path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(os.path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loader
    poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print("==> Calculating average elbow joint coordinates...")
    left_elbow_idx = 3  # Assuming left elbow index, adjust as needed
    right_elbow_idx = 4  # Assuming right elbow index, adjust as needed

    left_elbows = []
    right_elbows = []

    # Loop through the validation data
    for i, (targets_3d, _, _) in enumerate(valid_loader):
        left_elbows.append(targets_3d[:, left_elbow_idx, :].cpu().numpy())
        right_elbows.append(targets_3d[:, right_elbow_idx, :].cpu().numpy())

    # Convert lists to arrays and calculate the average
    left_elbows = np.concatenate(left_elbows, axis=0)
    right_elbows = np.concatenate(right_elbows, axis=0)

    avg_left_elbow = np.mean(left_elbows, axis=0)
    avg_right_elbow = np.mean(right_elbows, axis=0)

    print(f"Average left elbow coordinates: {avg_left_elbow}")
    print(f"Average right elbow coordinates: {avg_right_elbow}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
