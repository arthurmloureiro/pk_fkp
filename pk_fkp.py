#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Using gaussian realizations to generate maps of galaxies and analize it with a FKP method
	
	Arthur E. M. Loureiro & Lucas F. Secco
			18/07/2014
			  IFUSP

"""
import numpy as np
import grid3D as gr
########################################
# Reading the input file and converting 
########################################
camb_file, cell_size, n_x, n_y, n_z, num_realiz, bias, num_bins = np.loadtxt('input.dat', dtype=str)
cell_size = float(cell_size); n_x=int(n_x); n_y=int(n_y); n_z=int(n_z); num_realiz=int(num_realiz); bias=float(bias) ; num_bins=int(num_bins);
######################
# Reading CAMB's file
######################
k_camb , Pk_camb = np.loadtxt(camb_file, unpack=True)	   
