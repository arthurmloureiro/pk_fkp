#! /usr/bin/env python
# -*- coding: utf-8 -*-
#################################################
# This is the input file for the pk_fkp.py pack #
#################################################

# CAMB's original Power Spectra file:
camb_file = "fid2_matterpower.dat"

# Size of the cell in Mpc*h^-1
cell_size = 20.00

# Number of cells in x,y,z (must be integer)
n_x,n_y,n_z = 64,64,64

# Number of Tracers:
n_tracers = 1

# Bias for each tracer:
bias = [1]

# Mean galaxy number / cell for each tracer:
n_bar0 = [5.]

#Number of realizations (must be integer)
num_realiz = 1

# map file's root:
file_root = "g_bias"
