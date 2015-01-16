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
n_tracers = 3

# Bias for each tracer:
bias = [1.,2.0,1.5]

# Mean galaxy number / cell for each tracer:
n_bar0 = [8.,10.,12.]

#Number of realizations (must be integer)
num_realiz = 1

################################################
# Which kind of realizations to vary? 
#(1) Gaussian + Poissonian (2) Poissonian Only
################################################
realiz_type = 1
