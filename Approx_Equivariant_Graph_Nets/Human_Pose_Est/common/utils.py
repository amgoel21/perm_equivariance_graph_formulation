from __future__ import absolute_import, division

import os
import torch
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result



def perm_2d(perm):
    # Turns a permutation into permutation of 2d orbits
    perm_2d=[0]*(len(perm))**2
    for i in range(len(perm)):
        for j in range(len(perm)):
            perm_2d[i*len(perm)+j]=len(perm)*perm[i]+perm[j]
    return perm_2d

def orbit_2d(constructors):
    """
    Args: 
    constructors: List of permutation constructors of equivariant group G

    Returns:
    List of lists of tuples (a,b)
    Each tuple corresponds to edge from node a to node b
    Each list is an orbit (meaning equivalent edges in 2-closure Graph)
    """
    n = constructors[0].size
    perms_2d=[]
    for c in constructors:
        perms_2d.append(Permutation(perm_2d(list(c))))
    new_group=PermutationGroup(perms_2d)
    orbits = new_group.orbits()
    altered_orbits = []
    for orbit in orbits:
        new_orbit = []
        for i in orbit:
            new_orbit.append((i//n,i%n))
        altered_orbits.append(new_orbit)
    return altered_orbits



def sparsify_corrected(constructors,max_norm_main=20, max_norm_nonmain=1, soft = False):
    """
    Creates a sparse graph representation based on main edges and orbit_2d categories.
    Categories with at least one main edge get max_norm=100, others get max_norm=1.

    Args:
        constructors: List of permutation constructors.
        n_nodes: Number of nodes in the graph.
        find_mainedges: Function that returns the set of main edges.
        max_norm_main: Maximum norm for embeddings of categories with main edges.
        max_norm_nonmain: Maximum norm for embeddings of categories without main edges.

    Returns:
        num_categories: Total number of unique edge categories.
        edges: List of (source, target) main edges.
        edge_categories: List of category indices for each main edge.
        category_norms: Dict mapping category indices to their max norm.
    """
    # Step 1: Find main edges
    main_edges = [(9,8),(8,9),(10,8),(8,10),(10,11),(11,10),(11,12),(12,11),(8,13),(13,8),(14,13),(13,14),(14,15),(15,14),(8,7),(7,8),(7,0),(0,7),(0,1),(0,4),(4,0),(1,0),(1,2),(2,3),(3,2),(2,1),(4,5),(5,6),(6,5),(5,4)]
    if not soft:
        max_norm_main = 100000000
        max_norm_nonmain = 0
        

    # Step 2: Compute categories using orbit_2d
    orbits = orbit_2d(constructors)
    num_categories = len(orbits)

    # Step 3: Assign max_norm based on edge type
    category_norms = {}
    edges = []
    edge_categories = []

    for cat_idx, orbit in enumerate(orbits):
        # Check if the category contains any main edges
        is_main_category = any(edge in main_edges for edge in orbit)
        max_norm = max_norm_main if is_main_category else max_norm_nonmain
        category_norms[cat_idx] = max_norm

        for edge in orbit:
            edges.append(edge)
            edge_categories.append(cat_idx)

    return num_categories, edges, edge_categories, category_norms




    
    
    
