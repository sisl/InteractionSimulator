# test_datautils.py

import pandas as pd
import intersim.utils as utils
import matplotlib.pyplot as plt

def main():

	# load a trackfile
	df = pd.read_csv(f'{utils.DATASET_DIR}/trackfiles/DR_USA_Roundabout_FT/vehicle_tracks_000.csv')

	stv = utils.df_to_stackedvehicletraj(df)
	states, _ = utils.SVT_to_stateactions(stv)

	T,nv5 = states.shape
	nv = nv5//5

	states = states.reshape(T,nv,5)

	for i in range(5):

		ti0 = int(stv.t0[i]/stv.dt)
		tie = ti0 + len(stv.s[i])
		statesi = states[:,i]
		statesi = statesi[ti0:tie-1]

		plt.subplot(4,1,1)
		plt.plot(statesi[:,0], statesi[:,1])
		plt.plot(statesi[0,0], statesi[0,1], '*')

		plt.subplot(4,1,2)
		plt.plot(statesi[:,2])

		plt.subplot(4,1,3)
		plt.plot(statesi[:,3])

		plt.subplot(4,1,4)
		plt.plot(statesi[:,4])

		plt.show()


if __name__ == '__main__':
	main()
