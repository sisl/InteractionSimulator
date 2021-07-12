# vehicletraj.py

import numpy as np
from numpy.polynomial.polynomial import Polynomial as P
from numpy.polynomial.polynomial import polyder
import torch
import random

class StackedVehicleTraj:

	def __init__(self, lengths, widths, t0, s, v, xpoly, ypoly, dt=0.1):
		"""
		Stacked vehicle trajectories as polynomials.
		Args:
			lengths (torch.tensor): (nv,) vehicle lengths
			widths (torch.tensor): (nv,) vehicle widths
			t0 (torch.tensor): (nv,) start times
			s (list of torch.tensor): list of nv path positions
			v (list of torch.tensor): list of nv path velocities
			xpoly (torch.tensor): (nv, K) polynomial coefficients of x(s)
			ypoly (torch.tensor): (nv, K) polynomial coefficients of y(s)
			dt (float): timestep
		"""

		self._lengths = lengths
		self._widths = widths

		self._nv = len(lengths)
		self._t0 = t0
		self._s = s
		self._v = v
		self._smax = torch.tensor([s_[-1] for s_ in s])
		self._v0 = torch.tensor([v_[0] for v_ in v])

		self._xpoly = xpoly.detach()
		self._ypoly = ypoly.detach()

		# compute dxpoly, dypoly, ddxpoly, ddypoly
		def difpoly(poly):
			poly = poly.detach().numpy()
			dpoly = torch.zeros(*poly.shape).type(xpoly.type())
			ddpoly = torch.zeros(*poly.shape).type(xpoly.type())
			for i, p in enumerate(poly):

				dpolser = polyder(p).copy()
				dpolser = torch.tensor(dpolser)
				dpoly[i,:-1] = dpolser

				ddpolser = polyder(p, 2).copy()
				ddpolser = torch.tensor(ddpolser)
				ddpoly[i,:-2] = ddpolser

			return dpoly, ddpoly

		self._dxpoly, self._ddxpoly = difpoly(xpoly)
		self._dypoly, self._ddypoly = difpoly(ypoly)

		self._dt = dt

		# compute min, max T
		minT = float(t0.min())
		minTind = int(minT/dt)
		maxTind = max([ len(s_) + int(t0_/dt) for (s_,t0_) in zip(s,t0) ])
		maxT = maxTind * dt
		Tind = maxTind - minTind
		T = maxT - minT

		self._minT = minT
		self._maxT = maxT
		self._minTind = minTind
		self._maxTind = maxTind
		self._Tind = Tind
		self._T = T

		# initial states
		state0 = torch.zeros(self._nv, 2) * np.nan
		for i in range(self._nv): 
			if self._t0[i] == minT:
				state0[i,0] = s[i][0]
				state0[i,1] = v[i][0]

		self._state0 = state0
		self._shuffled = False

	def shuffle_tracks(self, seed: int = None):
		self._shuffled = True
		if seed:
			random.seed(seed)
		self._shuffle_idx = list(range(self._nv))
		random.shuffle(self._shuffle_idx)

		# update stored values based on self._shuffle_idx
		self._lengths = self._lengths[self._shuffle_idx]
		self._widths = self._widths[self._shuffle_idx]
		self._s = [self._s[i] for i in self._shuffle_idx]
		self._v = [self._v[i] for i in self._shuffle_idx]
		self._t0 = self._t0 # [self._shuffle_idx] DO NOT SHUFFLE TIME
		# TODO: make shuffled start_times random
		self._v0 = self._v0[self._shuffle_idx]
		self._smax = self._smax[self._shuffle_idx]
		
		self._xpoly = self._xpoly[self._shuffle_idx]
		self._ypoly = self._ypoly[self._shuffle_idx]
		self._dxpoly = self._dxpoly[self._shuffle_idx]
		self._dypoly = self._dypoly[self._shuffle_idx]
		self._ddxpoly = self._ddxpoly[self._shuffle_idx]
		self._ddypoly = self._ddypoly[self._shuffle_idx]

	@property
	def state0(self):
		return self._state0

	@property
	def polydeg(self):
		return self._xpoly.shape[-1]-1
	
	@property
	def xpoly(self):
		return self._xpoly
			
	@property
	def ypoly(self):
		return self._ypoly
	
	@property
	def dxpoly(self):
		return self._dxpoly
	
	@property
	def dypoly(self):
		return self._dypoly
	
	@property
	def ddxpoly(self):
		return self._ddxpoly
	
	@property
	def ddypoly(self):
		return self._ddypoly
		
	@property
	def T(self):
		return self._T
	
	@property
	def Tind(self):
		return self._Tind

	@property
	def minT(self):
		return self._minT

	@property
	def minTind(self):
		return self._minTind
	
	@property
	def s(self):
		return self._s

	@property
	def v(self):
		return self._v
	
	@property
	def v0(self):
		return self._v0

	@property
	def smax(self):
		return self._smax

	@property
	def t0(self):
		return self._t0

	@property
	def nv(self):
		return self._nv

	@property
	def dt(self):
		return self._dt

	@property
	def lengths(self):
		return self._lengths
	
	@property
	def widths(self):
		return self._widths
	
	
	
	