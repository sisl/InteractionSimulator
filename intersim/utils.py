# utils.py

import pandas as pd
import numpy as np
import torch
from intersim.vehicletraj import StackedVehicleTraj
import warnings
warnings.simplefilter('ignore', np.RankWarning)

import os
opj = os.path.join

import intersim
DATASET_BASE = os.path.normpath(os.path.join(os.path.dirname(intersim.__file__), '..'))

LOCATIONS = [
    'DR_USA_Roundabout_FT',
    'DR_CHN_Roundabout_LN',
    'DR_DEU_Roundabout_OF',
    'DR_USA_Roundabout_EP',
    'DR_USA_Roundabout_SR'
]
MAX_TRACKS=5
def get_map_path(loc: int = 0, base: str = DATASET_BASE) -> str:
    """
    Get path to .osm map file from location index
    Args:
        loc (int): location index
        base (str): base path
    Returns:
        osm (str): path to .osm map file
    """
    assert loc >= 0 and loc < len(LOCATIONS), "Invalid location index {} not in [0,{}]".format(loc,len(LOCATIONS)-1)
    return opj(base, 'datasets','maps',LOCATIONS[loc]+'.osm')

def get_svt(loc: int = 0, track: int = 0, base: str = DATASET_BASE, deg=20):
    """
    Load stacked vehicle trajectory from location and track indices
    Args:
        loc (int): location index
        track (int): track index
        base (str): base path
        deg (int): polynomial degree for tracks
    Returns:
        svt (StackedVehicleTraj): stacked vehicle traj to base trajectories off of
        path (str): path to data used
    """
    assert loc >= 0 and loc < len(LOCATIONS), "Invalid location index {} not in [0,{}]".format(loc,len(LOCATIONS)-1)
    assert track >= 0 and track < MAX_TRACKS, "Invalid location index {} not in [0,{}]".format(track,MAX_TRACKS-1)
    path = opj(base, 'datasets','trackfiles',LOCATIONS[loc],'vehicle_tracks_%03i.csv'%(track))
    df = pd.read_csv(path)
    stv = df_to_stackedvehicletraj(df, deg=deg)
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

def horner_scheme(x, poly):
    """
    Use Horner scheme to evaluate polynomial

    Args:
        x (torch.tensor): (nv, nsteps) path coordinates where to evaluate polynomial
        poly (torch.tensor): (nv, deg) polynomial coefficients for all vehicles

    Returns:
        r (torch.tensor): (nv, nsteps) polynomial evaluated for all vehicles at all path coordinates
    """
    deg = poly.shape[-1]
    nsteps = x.shape[-1]
    r = poly[:, -1:].type(x.dtype).repeat(1, nsteps)
    for i in range(2, deg+1):
        r *= x
        r += poly[:, -i:1-i]
    return r

def df_to_stackedvehicletraj(df, deg=20):
    """
    Convert a vehicle_tracks dataframe to a StackedVehicleTraj
    Args:
        df (pd.DataFrame): vehicle_tracks dataframe
        deg (int): polynomial degree for tracks
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
        v = torch.tensor(v).type(torch.get_default_dtype())
        s = torch.tensor(s).type(torch.get_default_dtype())

        t0list.append(t0)
        slist.append(s)
        xlist.append(x)
        ylist.append(y)
        vlist.append(v)

    t0 = torch.tensor(t0list)
    lengths = torch.tensor(lengths)
    widths = torch.tensor(widths)

    xpoly, ypoly = polyfit_sxy(slist,xlist,ylist, deg=deg)

    dt = df_.timestamp_ms.diff().mean()/1000

    return StackedVehicleTraj(lengths, widths, t0, slist, vlist, xpoly, ypoly, dt=dt)


def polyfit_sxy(s, x, y, deg=20):
    """
    Map vehicle trajectories to xpoly, ypoly using np.polyfit
    Args:
        s (list of torch.tensor): list of nv path positions
        x (list of torch.tensor): list of nv path xs
        y (list of torch.tensor): list of nv path ys
        deg (int): polynomial degree for tracks
    Returns:
        xpoly (torch.tensor): (nv, deg+1) coeffs of x(s)
        ypoly (torch.tensor): (nv, deg+1) coeffs of y(s)
    """
    nv = len(s)
    xpoly = torch.zeros(nv, deg + 1)
    ypoly = torch.zeros(nv, deg + 1)
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
        simstates (torch.tensor): (T, nv, 5) states
    """

    nv = xpoly.shape[0]
    deg = xpoly.shape[-1] - 1
    T = s.shape[0]

    simstates = torch.ones(T, nv, 5) * np.nan
    expand_sims = powerseries(s.type(torch.float64), deg)

    x = (xpoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())
    dxds = (dxpoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())
    ddxds = (ddxpoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())
    y = (ypoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())
    dyds = (dypoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())
    ddyds = (ddypoly.unsqueeze(0)*expand_sims).sum(dim=-1).type(torch.get_default_dtype())

    psi = torch.atan2(dyds,dxds)

    den = dyds ** 2 + dxds ** 2
    psidot = (1./den) * (-dxds * ddyds + dyds * ddxds)


    simstates[...,0] = x
    simstates[...,1] = y
    simstates[...,2] = sdot
    simstates[...,3] = psi
    simstates[...,4] = psidot

    return simstates

def ssdot_to_simactions(s, sdot, dt=0.1):
    """
    Maps s, sdot to accelerations taken in between.
    Args:
        s (torch.tensor): (T,nv) arc lengths
        sdot (torch.tensor): (T,nv) velocities
    Returns:
        simactions (torch.tensor): (T-1, nv, 1) actions, taken in between states
    """
    # sdot(t+1) = sdot(t) + dt * a(t)
    # s(t+1) = s(t) +  0.5 * dt * (sdot(t+1) + sdot(t)) = s(t) + dt * sdot(t) + 0.5 * dt^2 * a(t)

    T, nv = s.shape
    # Get a from s, ignore sdot, except for sdot(0) (this is what simulator does anyway) 
    # TODO: filter accelerations better (e.g. states and velocities rather than just states)
    # TODO: vectorize 
    adj_sdot = sdot.clone()
    simactions = torch.zeros(T-1, nv, 1) # * np.nan
    for car in range(nv):
        for t in range(T-1):
            if np.isnan(s[t,car]):
                continue
            elif np.isnan(s[t+1,car]):
                a = 0.
            else:
                a = 2 * (s[t+1,car] - s[t,car] - (dt * adj_sdot[t,car])) / (dt**2)
            if abs(a) > 10:
                pass
                # set trace for future debug
                # import pdb
                # pdb.set_trace()
            simactions[t,car,0] = a
            adj_sdot[t+1,car] = adj_sdot[t,car] + dt * simactions[t,car,0]
    return simactions

def SVT_to_stateactions(svt: StackedVehicleTraj):
    """
    Converts a StackedVehicleTrajectory into Simulator state and actions
    Args:
        svt (StackedVehicleTraj)
    Returns:
        sim_projected_states (torch.tensor): (T, nv, 5) states
        sim_actions (torch.tensor): (T-1, nv, 1) actions
    """
    sim_projected_states = ssdot_to_simstates(svt.simstate[...,0], svt.simstate[...,1], 
                                svt.xpoly, svt.dxpoly, svt.ddxpoly,
                                svt.ypoly, svt.dypoly, svt.ddypoly)
    sim_actions = ssdot_to_simactions(svt.simstate[...,0], svt.simstate[...,1])
    return sim_projected_states, sim_actions










