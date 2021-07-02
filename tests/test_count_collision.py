#
# test_count_collision.py
#

import torch
import numpy as np
from intersim.collisions import count_collisions, states_to_polygons
import matplotlib.pyplot as plt

def test():

	torch.manual_seed(1)

	x = torch.randn(10,10,5) * 2.
	x[:,7] = np.nan

	lengths = torch.ones(10) * 1.
	widths = torch.ones(10) * 0.5

	ncols = count_collisions(x.reshape(10,-1), lengths, widths)

	print(ncols)
	# seems like T has 4 collisions, lets visualize and check

	pT = states_to_polygons(x[-1], lengths, widths)
	nni = ~torch.isnan(x[-1,:,0])
	pxT = x[-1,nni,0]
	pyT = x[-1,nni,1]
	psiT = x[-1,nni,3]

	if True:

		for p,pxT_, pyT_, psiT_ in zip(pT,pxT,pyT,psiT):
			x,y = p.exterior.xy
			plt.plot(x, y)
			plt.text(pxT_, pyT_, '%.0f'%float(psiT_/np.pi * 180))

		plt.show()


	# yup.

if __name__ == '__main__':
	test()