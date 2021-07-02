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



def count_collisions(x, lengths, widths):
    """
    Count collisions in a trajectory.
    Args:
        x (torch.tensor): (T,nv*5) vehicle state
        legnths (torch.tensor): (nv,) vehicle lengths
        widths (torch.tensor): (nv,) vehicle lengths
    Returns
        ncols (int): number of collisions
    """

    T,_ = x.shape
    x = x.reshape(T, -1, 5)
    nv0 = x.shape[1]
    cols = torch.zeros((nv0, nv0))
    for t in range(T):
        polys_t, nni_t = states_to_polygons(x[t], lengths, widths)
        nv = len(polys_t)
        for i in range(1,nv):
            for j in range(i):
                if polys_t[i].intersects(polys_t[j]):
                    cols[nni_t[i],nni_t[j]] += 1
                    
    cols = cols + cols.T # make symmetric 
    ncols = np.count_nonzero(cols) / 2 # divide by 2 since symmetric
    return ncols
