#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Using gaussian realizations to generate maps of galaxies and analize it with a FKP method
	
							Arthur E. M. Loureiro 
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
camb_file, cell_size, n_x, n_y, n_z, num_realiz, bias1, bias2, num_bins, n_bar1,n_bar2, realiz_type = np.loadtxt('input.dat', dtype=str)
cell_size = float(cell_size); n_x=int(n_x); n_y=int(n_y); n_z=int(n_z); num_realiz=int(num_realiz); bias1=float(bias1) ; num_bins=int(num_bins); realiz_type = int(realiz_type); n_bar1 = float(n_bar1); bias2 = float(bias2); n_bar2 = float(n_bar2); 

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
L_max = np.sqrt(L_x*L_x + L_y*L_y + L_z*L_z)								 # Maximum scale
#L_min = np.sqrt(3*cell_size**2.)
box_vol = L_x*L_y*L_z								   						 # Box's volume
print("Generating the k-space Grid...\n")
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)					  				 # generates the grid
grid_bins = gr.grid3d(num_bins, num_bins, num_bins, L_x,L_y,L_z)		     # generates the bins grid
										    						 		 # multiplying the grid for the cell_size will give us a grid in physical units 

k_min = np.min(k_camb[1:])
k_max = (2.*np.pi)/cell_size
k_step = (1./L_max)*(1./1.1)													# the 1/3. is Raul's idea


k_r = np.arange(k_min,3.*k_max,k_step)											# the 3*k_max is Raul's idea

######################################
# Finding Camb's Correlation Function
######################################
print("Finding the Correlation Function...\n")
r_max = (np.pi)/np.min(k_r[1:])*(0.3)										# Raul's ....
r_step = 1./np.max(grid.grid_k)*(2./3.)										# (2./3.) also Raul's idea
r_k=1.0*np.arange(0.0001,r_max,r_step)                                            

dk_r=np.diff(k_r)                                           			     # makes the diff between k and k + dk
dk_r=np.append(dk_r,[0.0])

krk=np.einsum('i,j',k_r,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_r*Pk_camb_interp(k_r)*np.exp(-1.0*np.power(k_r/(2.*k_max),6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)
integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)       		     # uses the trace in the r axis to make the integral

pl.figure()
pl.grid(1)
#pl.xscale("log")
pl.plot(r_k,(r_k*r_k)*corr_ln)

corr_g = np.log(1.+corr_ln) 	

#sys.exit()			                                     # Gaussian Correl. Func.
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
terms = np.einsum('i,j', drCorr,km1)
integrando2 = sinrk2*terms

Pk_gauss = 4.27*np.pi*np.sum(integrando2, axis=0)				# o fator na frente é originalmente 4!
Pk_gauss[0] = Pk_camb[1]

Pk_gauss_interp = interpolate.UnivariateSpline(k_r,Pk_gauss)	

final = clock()
time = final-init
print "time = " + str(time)

pl.figure()
pl.ylim(0.,10E6)
pl.loglog() 
pl.plot(k_camb, Pk_camb_interp(k_camb))
pl.plot(k_camb, Pk_camb_interp(k_camb)*np.exp(-1.0*np.power(k_camb/(2.*k_max),6.0)))
pl.plot(k_r, Pk_gauss_interp(k_r),'--' ,linewidth=2.5)
pl.axvline(x=np.max(grid.grid_k), linewidth=2., color='r')
pl.axvline(x=np.min(grid.grid_k[grid.grid_k!=0.0]), linewidth=2., color='g')
pl.plot(k_r,Pk_gauss)
pl.show()

sys.exit()



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
M = np.asarray([heav_vec(k_bar[a+1]-grid.grid_k[:,:,:])*heav_vec(grid.grid_k[:,:,:]-k_bar[a])for a in range(len(k_bar)-1)])
PN1 = np.zeros((len(k_bar[1:]), num_realiz))
PN2 = np.zeros((len(k_bar[1:]), num_realiz))
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
		delta_xr1 = delta_x_ln(delta_xr_g, var_gr,bias1)
		delta_xr2 = delta_x_ln(delta_xr_g, var_gr,bias2)
		#######################
		#poissonian realization
		#######################
		N_r1 = np.random.poisson(n_bar1*(1.+delta_xr1)*(cell_size**3.))			     # This is the final galaxy Map
		N_r2 = np.random.poisson(n_bar2*(1.+delta_xr2)*(cell_size**3.))
		delta_gg_r1 = (N_r1 - np.mean(N_r1))/np.mean(N_r1)
		delta_gg_r2 = (N_r2 - np.mean(N_r2))/np.mean(N_r2)
		delta_gg_k1 = (box_vol/(n_x*n_y*n_y))*np.fft.fftn(delta_gg_r1.real)
		delta_gg_k2 = (box_vol/(n_x*n_y*n_y))*np.fft.fftn(delta_gg_r2.real)
		P_gg1 = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k1, np.conj(delta_gg_k1))/(np.einsum("aijl->a", M)*box_vol)
		P_gg2 = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k2, np.conj(delta_gg_k2))/(np.einsum("aijl->a", M)*box_vol)
		PN1[:,m] = P_gg1.real
		PN2[:,m] = P_gg2.real
		
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
	delta_xr1 = delta_x_ln(delta_xr_g, var_gr,bias1)
	delta_xr2 = delta_x_ln(delta_xr_g, var_gr,bias2)
	for m in range(num_realiz):
		#######################
		#poissonian realization
		#######################
		N_r1 = np.random.poisson(n_bar1*(1.+delta_xr1)*(cell_size**3.))			     # This is the final galaxy Map
		N_r2 = np.random.poisson(n_bar2*(1.+delta_xr2)*(cell_size**3.))
		delta_gg_r1 = (N_r1 - np.mean(N_r1))/np.mean(N_r1)
		delta_gg_r2 = (N_r2 - np.mean(N_r2))/np.mean(N_r2)
		delta_gg_k1 = (box_vol/(n_x*n_y*n_z))*np.fft.fftn(delta_gg_r1.real)
		delta_gg_k2 = (box_vol/(n_x*n_y*n_z))*np.fft.fftn(delta_gg_r2.real)
		P_gg1 = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k1, np.conj(delta_gg_k1))/(np.einsum("aijl->a", M)*box_vol)
		P_gg2 = np.einsum("aijl,ijl,ijl->a", M, delta_gg_k2, np.conj(delta_gg_k2))/(np.einsum("aijl->a", M)*box_vol)
		PN1[:,m] = P_gg1.real
		PN2[:,m] = P_gg2.real
	print "\nDone.\n" 
else:
	print "Error, invalid option for realization's type \n"
	sys.exit(-1)
error_bar1 = np.zeros(len(k_bar[1:]))
error_bar2 = np.zeros(len(k_bar[1:]))
PN_mean1 = np.zeros(len(k_bar[1:]))
PN_mean2 =  np.zeros(len(k_bar[1:]))
for i in range(len(k_bar[1:])):
        PN_mean1[i] = np.mean(PN1[i,:])
        PN_mean2[i] = np.mean(PN2[i,:])
        error_bar1[i] = np.std(PN1[i,:])/np.mean(PN1[i,:])
        error_bar2[i] = np.std(PN2[i,:])/np.mean(PN2[i,:])


#sys.exit(-1)
final = clock()
print "time = " + str(final - inicial)		
pl.figure()
pl.loglog()
pl.errorbar(k_bar[1:], PN_mean1, yerr=error_bar1)
pl.plot(k_camb,Pk_camb)
pl.errorbar(k_bar[1:], PN_mean2, yerr=error_bar2)
pl.figure()
pl.imshow(N_r1[0], cmap=cm.jet)
pl.show()















