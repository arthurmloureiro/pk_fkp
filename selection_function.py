########################################
# This is the selection function module
########################################
import numpy as np
def selection_func(gridr,a_,b_):
	"""
	number of parameters = 2
	"""
	return a_*np.exp(-b_*gridr)

