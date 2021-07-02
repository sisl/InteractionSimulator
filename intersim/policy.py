# policy

import torch

class Policy:

	def compute_action(self, x):
		"""
		Abstract policy computing accelerations for each vehicle.
		Args:
			x (torch.tensor): (*, nv*5) vehicle states
		Returns:
			a (torch.tensor): (*,nv,1) actions
		"""
		raise NotImplementedError

	def __call__(self, x):
		"""
		Abstract policy computing accelerations for each vehicle.
		Args:
			x (torch.tensor): (*, nv*5) vehicle states
		Returns:
			a (torch.tensor): (*,nv,1) actions
		"""
		nv = x.shape[-1] // 5
		if x.ndim == 1:
			return self.compute_action(x.unsqueeze(0)).reshape(nv,1)

		else:
			return self.compute_action(x)