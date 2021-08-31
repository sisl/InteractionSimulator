# test_datautils.py

import pandas as pd
from intersim import utils

def test_datautils():

	# load a trackfile
	df = pd.read_csv(f'{utils.DATASET_DIR}/trackfiles/DR_USA_Roundabout_FT/vehicle_tracks_000.csv')

	svt = utils.df_to_stackedvehicletraj(df)
	states, actions = utils.SVT_to_stateactions(svt)

	T,nv,_ = states.shape

	for i in range(5):

		ti0 = int(svt.t0[i]/svt.dt)
		tie = ti0 + len(svt.s[i])
		statesi = states[:,i]
		statesi = statesi[ti0:tie-1]

		#plt.subplot(4,1,1)
		#plt.plot(statesi[:,0], statesi[:,1])
		#plt.plot(statesi[0,0], statesi[0,1], '*')

		#plt.subplot(4,1,2)
		#plt.plot(statesi[:,2])

		#plt.subplot(4,1,3)
		#plt.plot(statesi[:,3])

		#plt.subplot(4,1,4)
		#plt.plot(statesi[:,4])

		#plt.show()


if __name__ == '__main__':
	test_datautils()
