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
from time import clock
from scipy import interpolate
import pylab as pl
from matplotlib import cm

init = clock()
#################################################
# Reading the input file and converting the data
#################################################
camb_file, cell_size, n_x, n_y, n_z, num_realiz, bias, num_bins, n_bar, realiz_type = np.loadtxt('input.dat', dtype=str)
cell_size = float(cell_size); n_x=int(n_x); n_y=int(n_y); n_z=int(n_z); num_realiz=int(num_realiz); bias=float(bias) ; num_bins=int(num_bins); realiz_type = int(realiz_type); n_bar = float(n_bar);

######################
# Reading CAMB's file
######################
k_camb , Pk_camb = np.loadtxt(camb_file, unpack=True)
k_camb = np.insert(k_camb,0,0.)						
Pk_camb = np.insert(Pk_camb,0,0.)
Pk_camb_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_camb)	     #interpolate camb's Power Spectrum

#######################
# Initial calculations
#######################
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size 		     # size of the box
box_vol = L_x*L_y*L_z								     # Box's volume
L_max = np.sqrt(L_x*L_x + L_y*L_y + L_z*L_z)	

print("Generating the k-space Grid...\n")
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)					     # generates the grid
grid_bins = gr.grid3d(num_bins, num_bins, num_bins, L_x,L_y,L_z)		     # generates the bins grid
										     # multiplying the grid for the cell_size will give us a grid in physical units 
###################################################
# This will be used to calculate the Gaussian P(k)
###################################################
k_min = np.min(k_camb[1:])									     
k_max = (2.*np.pi)/cell_size										     
k_step = 1./L_max*(1./1.1)
k_r = np.arange(k_min,3.*k_max,k_step)

######################################
# Finding Camb's Correlation Function
######################################
print("Finding the Correlation Function...\n")
r_max = (np.pi)/np.min(k_r[1:])*(0.3)										#there's no acctual reason to choose this 0.3
r_step = 1./np.max(grid.grid_k)*(2./3.)
r_k=1.0*np.arange(1.,r_max,r_step)

dk_r=np.diff(k_r)                                           			     # makes the diff between k and k + dk
dk_r=np.append(dk_r,[0.0])

krk=np.einsum('i,j',k_r,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_r*Pk_camb_interp(k_r)*np.exp(-1.0*np.power(k_r/(2.*k_max),6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)
integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)       		     # uses the trace in the r axis to make the integral
corr_g = np.log(1.+corr_ln) 	
						                                     # Gaussian Correl. Func.
######################################
# Finding the gaussian power spectrum
######################################
print("Calculating the Gaussian P(k)...\n Any Error Warning here is expected. \n")
dr = np.diff(r_k)
dr = np.append(dr,[0.0])
rkr = np.outer(r_k,k_r)
sinrk2 = np.sin(rkr)
drCorr = dr*r_k*corr_g
km1 = np.power(k_r,-1.)
terms = np.outer(drCorr,km1)
integrando2 = sinrk2*terms

Pk_gauss = 4.*np.pi*np.sum(integrando2, axis=0)								
Pk_gauss[0] = Pk_camb[1]

Pk_gauss_interp = interpolate.UnivariateSpline(k_r,Pk_gauss)	
final = clock()
time = final-init
print "P_g(k) time = " + str(time)

###############################################################
# Generating the P(K) grid using the gaussian interpolated Pkg
###############################################################
print("\nCalculating the P(k)-Grid...\n")
Pkg_vec = np.vectorize(Pk_gauss_interp)
p_matrix = Pkg_vec(grid.grid_k)
p_matrix[0][0][0] = 1. 						     # Needs to be 1.
###########################################
# Defining the p.d.fs and useful functions
###########################################
def A_k(P_):									 
	#################################################################################
	# The Gaussian Amplitude
	# Zero Medium and STD=SQRT(2*P(k)*Volume)
	# It must have the 2 factor to take the complex part into account after the iFFT
	#################################################################################
	return np.random.normal(0.0,np.sqrt(2.*P_*box_vol))			    
										 						   
def phi_k(P_): 									     	
	######################
	# Random regular phase
	######################
	return (np.random.random(len(P_)))*2.*np.pi	

def delta_k_g(P_):								     
	########################################	
	# The density contrast in Fourier Space
	########################################
	return A_k(P_)*np.exp(1j*phi_k(P_))		

def delta_x_ln(d_,sigma_,bias_):
	###############################
	# The log-normal density field
	###############################
	return np.exp(bias_*d_ - ((bias_**2.)*(sigma_))/2.0) -1.

def heav(x):                                                    
	#####################
	# Heaviside Function
	#####################
	if x==0:
		return 0.5
	return 0 if x<0 else 1
heav_vec = np.vectorize(heav)					         #heaviside vetorizada


################################################################
# FFT Loops for Gaussian and Gaussian + Poissonian Realizations
################################################################
k_bar = np.arange(0,num_bins,1)*(np.max(grid.grid_k)/num_bins)
inicial = clock()
#file = open('supergrid2.dat','w')
if realiz_type == 1:
	print "Doing both Gaussian + Poissonian realizations... \n"
	for m in range(num_realiz):
		#########################
		# gaussian density field
		#########################
		delta_x_gaus = ((delta_k_g(p_matrix).size)/box_vol)*np.fft.ifftn(delta_k_g(p_matrix))	#the iFFT
		var_gr = np.var(delta_x_gaus.real)
		var_gi = np.var(delta_x_gaus.imag)
		delta_xr_g = delta_x_gaus.real
		delta_xi_g = delta_x_gaus.imag
		###########################
		# Log-Normal Density Field
		###########################
		delta_xr = delta_x_ln(delta_xr_g, var_gr,bias)
		delta_xi = delta_x_ln(delta_xi_g, var_gi,bias)
		#######################
		#poissonian realization
		#######################
		N_r = np.random.poisson(n_bar*(1.+delta_xr)*(cell_size**3.))			     # This is the final galaxy Map
		N_i = np.random.poisson(n_bar*(1.+delta_xi)*(cell_size**3.))
		##########################################
		#$%%$ AQUI SEGUE O CÓDIGO PARA O FKP $%%$#
		##########################################
		
	print "\nDone.\n"
elif realiz_type == 2:
	print "Doing Poissonian realizations only \n"
	#########################
	# gaussian density field
	#########################
	delta_x_gaus = ((delta_k_g(p_matrix).size)/box_vol)*np.fft.ifftn(delta_k_g(p_matrix))	#the iFFT
	var_gr = np.var(delta_x_gaus.real)
	var_gi = np.var(delta_x_gaus.imag)
	delta_xr_g = delta_x_gaus.real
	delta_xi_g = delta_x_gaus.imag
	###########################
	# Log-Normal Density Field
	###########################
	delta_xr = delta_x_ln(delta_xr_g, var_gr,bias)
	delta_xi = delta_x_ln(delta_xi_g, var_gi,bias)
	for m in range(num_realiz):
		#######################
		#poissonian realization
		#######################
		N_r = np.random.poisson(n_bar*(1.+delta_xr)*(cell_size**3.))     # This is the final galaxy Map
		N_i = np.random.poisson(n_bar*(1.+delta_xi)*(cell_size**3.))
#		n_bar0_new = np.mean(N_r)
		##########################################
		#$%%$ AQUI SEGUE O CÓDIGO PARA O FKP $%%$#
		##########################################
	print "\nDone.\n" 
else:
	print "Error, invalid option for realization's type \n"
	sys.exit(-1)
#file.close()

final = clock()
print "time = " + str(final - inicial)		
pl.figure()
pl.imshow(N_r[0], cmap=cm.jet)
pl.show()




