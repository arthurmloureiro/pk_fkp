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
import sys
from scipy import interpolate


########################################
# Reading the input file and converting 
########################################
camb_file, cell_size, n_x, n_y, n_z, num_realiz, bias, num_bins = np.loadtxt('input.dat', dtype=str)
cell_size = float(cell_size); n_x=int(n_x); n_y=int(n_y); n_z=int(n_z); num_realiz=int(num_realiz); bias=float(bias) ; num_bins=int(num_bins);

######################
# Reading CAMB's file
######################
k_camb , Pk_camb = np.loadtxt(camb_file, unpack=True)
k_camb = np.insert(k_camb,0,0.)						
Pk_camb = np.insert(Pk_camb,0,0.)		
Pk_camb_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_camb)		     #interpolate camb's Power Spectrum

#######################
# Initial calculations
#######################
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size 		     # size of the box
box_vol = L_x*L_y*L_z								     # Box's volume
print("Generating the k-space Grid...\n")
k_grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)					     # generates the k-grid

######################################
# Finding Camb's Correlation Function
######################################
print("Finding the Correlation Function...\n")
r_k=1.0*np.linspace(0.5,200.5,201)                                                   # r vector goes from 0.5 to 201 h^-1 MPc
dk_r=np.diff(k_camb)                                           			     # makes the diff between k and k + dk
dk_r=np.append(dk_r,[0.0])

krk=np.einsum('i,j',k_camb,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_camb*Pk_camb*np.exp(-1.0*np.power(k_camb/0.8,6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)
integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)       		     # uses the trace in the r axis to make the integral

corr_g = np.log(1.+corr_ln) 							     # Gaussian Correl. Func.
######################################
# Finding the gaussian power spectrum
######################################
print("Calculating the Gaussian P(k)...\n")
dr = np.diff(r_k)
dr = np.append(dr,[0.0])
rkr = np.einsum('i,j', r_k,k_camb)
sinrk2 = np.sin(rkr)
drCorr = dr*r_k*corr_g
km1 = np.power(k_camb,-1.)
terms = np.einsum('i,j', drCorr,km1)
integrando2 = sinrk2*terms

Pk_gauss = 4.0*np.pi*np.sum(integrando2, axis=0)
Pk_gauss[0] = 0.0
Pk_gauss_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_gauss)	

###############################################################
# Generating the P(K) grid using the gaussian interpolated Pkg
###############################################################

#Seguir de onde este comentário para em pk.py





