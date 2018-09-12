import numpy as np

exp_data = np.loadtxt("YbYVO_burnscan_B=1T_1.65K_holeburningexample_RAW.txt", delimiter=",")
exp_data[:, 0] = (exp_data[:, 0]+0.172)
np.savetxt("YbYVO_burnscan_B=1T_1.65K_holeburningexample.txt", exp_data, delimiter=",", fmt=("%.4e", "%.4e", "%.4e", "%.4e"))