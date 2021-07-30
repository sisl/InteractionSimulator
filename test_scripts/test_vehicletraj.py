# test_vehicletraj.py

from intersim.vehicletraj import StackedVehicleTraj
import torch

def main():

	t0 = torch.rand(50)
	s = [torch.arange(10).float() for i in range(50)]
	v = [torch.ones(10).float() for i in range(50)]
	xpoly = torch.randn(50, 70)
	ypoly = torch.randn(50, 70)
	lengths = torch.rand(50)
	widths = torch.rand(50)

	StackedVehicleTraj(lengths, widths, t0, s, v, xpoly, ypoly)
if __name__ == '__main__':
	main()
