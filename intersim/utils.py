# utils.py

import pandas as pd
import numpy as np
import torch
from intersim.vehicletraj import StackedVehicleTraj
import warnings
warnings.simplefilter('ignore', np.RankWarning)

torch.set_default_dtype(torch.float64)
import os
opj = os.path.join

LOCATIONS = [
    'DR_USA_Roundabout_FT',
    'DR_CHN_Roundabout_LN',
    'DR_DEU_Roundabout_OF',
    'DR_USA_Roundabout_EP',
    'DR_USA_Roundabout_SR'
]
MAX_TRACKS=5
def get_map_path(loc: int) -> str:
    """
    Get path to .osm map file from location index
    Args:
        loc (int): location index
    Returns:
        osm (str): path to .osm map file
    """
    assert loc >= 0 and loc < len(LOCATIONS), "Invalid location index {} not in [0,{}]".format(loc,len(LOCATIONS)-1)
    return opj('datasets','maps',LOCATIONS[loc]+'.osm')

def get_svt(loc: int, track: int):
    """
    Load stacked vehicle trajectory from location and track indices
    Args:
        loc (int): location index
        track (int): track index
    Returns:
        svt (StackedVehicleTraj): stacked vehicle traj to base trajectories off of
        path (str): path to data used
    """
    assert loc >= 0 and loc < len(LOCATIONS), "Invalid location index {} not in [0,{}]".format(loc,len(LOCATIONS)-1)
    assert track >= 0 and track < MAX_TRACKS, "Invalid location index {} not in [0,{}]".format(track,MAX_TRACKS-1)
    path = opj('datasets','trackfiles',LOCATIONS[loc],'vehicle_tracks_%03i.csv'%(track))
    df = pd.read_csv(path)
    stv = df_to_stackedvehicletraj(df)
    return stv, path

def to_circle(x):
        """
        Casts x (in rad) to [-pi, pi)
        
        Args:
            x (torch.tensor): (*) input angles (radians)
            
        Returns:
            y (torch.tensor): (*) x cast to [-pi, pi)
        """
        y = torch.remainder(x + np.pi, 2*np.pi) - np.pi
        return y

def powerseries(x, deg):

    return torch.stack([x**i for i in range(deg+1)],dim=-1)

def df_to_stackedvehicletraj(df):
    """
    Convert a vehicle_tracks dataframe to a StackedVehicleTraj
    Args:
        df (pd.DataFrame): vehicle_tracks dataframe
    Returns:
        svt (StackedVehicleTraj): stacked vehicle traj
    """

    lengths = []
    widths = []

    t0list = []
    slist = []
    xlist = []
    ylist = []
    vlist = []

    for tid in df.track_id.unique():

        df_ = df.loc[df.track_id == tid]

        lengths.append(df_.length.values[0])
        widths.append(df_.width.values[0])

        assert len(df_)-1 == (df_.frame_id.max() - df_.frame_id.min()), 'not contigous?'

        t0 = df_.timestamp_ms.min()/1000
        x = df_.x.values[:-1]
        y = df_.y.values[:-1]
        v = (df_.vx.values[:-1] ** 2 + df_.vy.values[:-1] ** 2) ** (1./2)

        dx = df_.x.diff().values[1:]
        dy = df_.y.diff().values[1:]
        ds = (dx ** 2 + dy ** 2) ** (1./2)

        s = np.cumsum(ds)

        # to tensors
        x = torch.tensor(x)
        y = torch.tensor(y)
        v = torch.tensor(v)
        s = torch.tensor(s)

        t0list.append(t0)
        slist.append(s)
        xlist.append(x)
        ylist.append(y)
        vlist.append(v)

    t0 = torch.tensor(t0list)
    lengths = torch.tensor(lengths)
    widths = torch.tensor(widths)

    xpoly, ypoly = polyfit_sxy(slist,xlist,ylist)

    dt = df_.timestamp_ms.diff().mean()/1000

    return StackedVehicleTraj(lengths, widths, t0, slist, vlist, xpoly, ypoly, dt=dt)


def polyfit_sxy(s, x, y, deg=20):
    """
    Map vehicle trajectories to xpoly, ypoly using np.polyfit
    Args:
        s (list of torch.tensor): list of nv path positions
        x (list of torch.tensor): list of nv path xs
        y (list of torch.tensor): list of nv path ys
    Returns:
        xpoly (torch.tensor): (nv, deg+1) coeffs of x(s)
        ypoly (torch.tensor): (nv, deg+1) coeffs of y(s)
    """
    nv = len(s)
    xpoly = torch.zeros(nv, deg + 1).type(torch.get_default_dtype())
    ypoly = torch.zeros(nv, deg + 1).type(torch.get_default_dtype())
    for i, (s_,x_,y_) in enumerate(zip(s,x,y)):
        s_ = s_.detach().numpy()
        x_ = x_.detach().numpy()
        y_ = y_.detach().numpy()
        pfx = np.polyfit(s_,x_,deg)[::-1].copy()
        pfy = np.polyfit(s_,y_,deg)[::-1].copy()
        xpoly[i,:] = torch.tensor(pfx)
        ypoly[i,:] = torch.tensor(pfy)

    return xpoly, ypoly

def ssdot_to_simstates(s, sdot, 
                      xpoly, dxpoly, ddxpoly, 
                      ypoly, dypoly, ddypoly):
    """
    Maps s, sdot to [x,y,v,psi,psidot] using poly coefficients.
    Args:
        s (torch.tensor): (T,nv) arc lengths
        sdot (torch.tensor): (T,nv) velocities
    Returns:
        simstates (torch.tensor): (T, nv * 5) states
    """

    nv = xpoly.shape[0]
    deg = xpoly.shape[-1] - 1
    T = s.shape[0]

    simstates = torch.ones(T, nv, 5) * np.nan
    expand_sims = powerseries(s, deg)

    x = (xpoly.unsqueeze(0)*expand_sims).sum(dim=-1)
    dxds = (dxpoly.unsqueeze(0)*expand_sims).sum(dim=-1)
    ddxds = (ddxpoly.unsqueeze(0)*expand_sims).sum(dim=-1)
    y = (ypoly.unsqueeze(0)*expand_sims).sum(dim=-1)
    dyds = (dypoly.unsqueeze(0)*expand_sims).sum(dim=-1)
    ddyds = (ddypoly.unsqueeze(0)*expand_sims).sum(dim=-1)

    psi = torch.atan2(dyds,dxds)

    den = dyds ** 2 + dxds ** 2
    psidot = (1./den) * (-dxds * ddyds + dyds * ddxds)


    simstates[...,0] = x
    simstates[...,1] = y
    simstates[...,2] = sdot
    simstates[...,3] = psi
    simstates[...,4] = psidot

    return simstates.reshape(T, -1)

def SVT_to_simstates(svt):
    """
    Converts a StackedVehicleTrajectory into Simulator states
    Args:
        svt (StackedVehicleTrajectory): stacked vehicle trajectory
    Returns:
        simstates (torch.tensor): (svt.Tind, svt.nv * 5) states
    """

    sims = torch.ones(svt.Tind, svt.nv) * np.nan
    simv = torch.ones(svt.Tind, svt.nv) * np.nan
    simstates = torch.ones(svt.Tind, svt.nv, 5) * np.nan

    for i in range(svt.nv):

        si = svt.s[i]
        vi = svt.v[i]
        ti = int(svt.t0[i]/svt.dt) - svt.minTind

        sims[ti:ti+len(si), i] = si
        simv[ti:ti+len(vi), i] = vi

    simstates = ssdot_to_simstates(sims, simv, 
                                svt.xpoly, svt.dxpoly, svt.ddxpoly,
                                svt.ypoly, svt.dypoly, svt.ddypoly)

    return simstates.reshape(svt.Tind, -1)










