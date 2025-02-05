from __future__ import absolute_import, division

import os
import torch
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation, PermutationGroup
from random import randrange
import math




def perm_2d(perm):
    # Turns a permutation into permutation of 2-tuples 
    perm_2d=[0]*(len(perm))**2

    for i in range(len(perm)):
        for j in range(len(perm)):
            perm_2d[i*len(perm)+j]=len(perm)*perm[i]+perm[j] # Each tuple is represented by integer

    return perm_2d

def orbit_2d(constructors):
    """
    Computes the 2D orbits of a set of permutation constructors.

    Parameters
    ----------
    constructors : list of Permutation objects
        The list of permutation constructors.

    Returns
    -------
    altered_orbits : list of lists of tuples
        The list of orbits, where each tuple in the list is a 2-tuple.
    """

    n = constructors[0].size

    perms_2d = []
    for c in constructors:
        perms_2d.append(Permutation(perm_2d(list(c))))

    new_group = PermutationGroup(perms_2d)
    orbits = new_group.orbits()

    altered_orbits = []
    for orbit in orbits:
        new_orbit = []
        for i in orbit:
            new_orbit.append((i//n, i%n))  # convert each integer into a 2D point
        altered_orbits.append(new_orbit)
        
    return altered_orbits


def sparsify(constructors, main_edges, max_norm_main=20, max_norm_nonmain=1,soft = False ):
    """
    Creates a sparse graph representation based on main edges and orbit_2d categories.
    

    Args:
        constructors: List of permutation constructors.
        main_edges: Set of main edges.

    Returns:
        num_categories: Total number of unique edge categories (2-tuple orbits)
        edges: List of (source, target) main edges.
        edge_categories: List of category indices for each main edge.
    """
    if not soft:
        max_norm_main = float('inf')
        max_norm_nonmain = 0
        

    #Compute categories using orbit_2d
    orbits = orbit_2d(constructors)
    num_categories = 0

    # Assign max_norm based on edge type
    edges = []
    edge_categories = []

    for cat_idx, orbit in enumerate(orbits):
        # Check if the category contains any main edges
        is_main_category = any(edge in main_edges for edge in orbit)
        if is_main_category:
            for edge in orbit:
                edges.append(edge)
                edge_categories.append(num_categories)
            num_categories+=1


    return num_categories, edges, edge_categories



def soft_sparsify(constructors, main_edges, max_norm_main=20, max_norm_nonmain=1,soft = False ):
    """
    Creates a (soft) sparse graph representation based on main edges and orbit_2d categories.
    Differentiates between main and non-main edges by edge norm
    

    Args:
        constructors: List of permutation constructors.
        main_edges: Set of main edges.
        max_norm_main: Maximum norm for embeddings of categories with main edges.
        max_norm_nonmain: Maximum norm for embeddings of categories without main edges.
        soft: Allow for non-main edges to exist, but with maximum norm equal to max_norm_nonmain (while main edges have max_norm_main)

    Returns:
        num_categories: Total number of unique edge categories (2-tuple orbits)
        edges: List of (source, target) main edges.
        edge_categories: List of category indices for each main edge.
        category_norms: Dict mapping category indices to their max norm.
    """
    if not soft:
        max_norm_main = float('inf')
        max_norm_nonmain = 0
        

    #Compute categories using orbit_2d
    orbits = orbit_2d(constructors)
    num_categories = len(orbits)

    # Assign max_norm based on edge type
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




    
    
    
