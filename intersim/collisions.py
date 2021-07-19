#
# collisions.py
#

import torch
import numpy as np
from shapely.geometry import Polygon


def states_to_polygons(x, lengths, widths):
    """
    Create a set of polygons representing each active vehicle.
    Args:
        x (torch.tensor): (nv, 5) vehicle states
        legnths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        vp (list of Polygon): (nv_active,) vehicle Polygons
        inds (torch.tensor): (n,) indices of active vehicles
    """

    nni = ~torch.isnan(x[:,0])
    x = x[nni]
    vp = []
    for  x_, li, wi in zip(x, lengths[nni], widths[nni]):
        px, py, v, psi, psidot = x_

        pxy = torch.stack([px, py])
        lon = torch.stack([psi.cos(), psi.sin()])
        lat = torch.stack([-psi.sin(), psi.cos()])
        
        ul = pxy + li/2. * lon + wi/2. * lat
        ur = pxy + li/2. * lon - wi/2. * lat
        ll = pxy - li/2. * lon + wi/2. * lat
        lr = pxy - li/2. * lon - wi/2. * lat
        
        corners = torch.stack([ll, lr, ur, ul]).detach().numpy()
        p = Polygon([*corners])
        vp.append(p)
    inds = torch.where(nni)[0]
    return vp, inds

def count_collisions_trajectory(x, lengths, widths):
    """
    Count collisions in a trajectory.
    Args:
        x (torch.tensor): (T,nv,5) vehicle states
        lengths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        ncols (int): number of collisions
    """

    T,nv0,_ = x.shape
    cols = torch.zeros((nv0, nv0))
    for t in range(T):
        cols += collision_matrix(x[t], lengths, widths)
    ncols = np.count_nonzero(cols) // 2 # divide by 2 since symmetric
    return ncols

def count_collisions(x, lengths, widths):
    """
    Count collisions in a single frame.
    Args:
        x (torch.tensor): (nv,5) vehicle states
        lengths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        ncols (int): number of collisions
    """

    cols = collision_matrix(x, lengths, widths)
    ncols = np.count_nonzero(cols) // 2 # divide by 2 since symmetric
    return ncols

def check_collisions_trajectory(x, lengths, widths):
    """
    Count collisions in a trajectory.
    Args:
        x (torch.tensor): (T,nv,5) vehicle states
        lengths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        cols torch.Tensor(bool): (T,) whether there is a collision at each time index
    """

    T,nv0,_ = x.shape
    cols = []
    for t in range(T):
        cols.append(check_collisions(x[t], lengths, widths))
    return torch.tensor(cols)
    
def check_collisions(x, lengths, widths):
    """
    Check whether there are collisions in a single frame.
    Args:
        x (torch.tensor): (nv,5) vehicle states
        lengths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        col (bool): True if there are collisions
    """
    return count_collisions(x, lengths, widths) > 0

def collision_matrix(x, lengths, widths):
    """
    Generate collision matrix at a state.
    Args:
        x (torch.tensor): (nv,5) vehicle state
        lengths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        c_mat (torch.tensor): (nv, nv) collision tensor 
            where [i,j] is True if i is colliding with j
    """
    nv0 = len(lengths)
    c_mat = torch.zeros(nv0, nv0)
    polys_t, nni_t = states_to_polygons(x, lengths, widths)
    nv = len(polys_t)
    for i in range(1,nv):
        for j in range(i):
            if polys_t[i].intersects(polys_t[j]):
                c_mat[nni_t[i],nni_t[j]] = 1
                c_mat[nni_t[j],nni_t[i]] = 1
    return c_mat