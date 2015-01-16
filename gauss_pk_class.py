#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Class to calculate the the gaussian power spectrum
	Arthur E. da Mota Loureiro (IFUSP)
	07/11/2014
"""
import numpy as np
from scipy import interpolate
import sys

class gauss_pk(object):
	"""
	This class is a part of the pk_fkp.
	It takes camb's power spectrum and transforms it in a gaussian P(k)
	The initial values are:
	k_camb, Pk_camb, the grid in k-space, cell_size and the maximum scale in the grid
	"""
	def __init__(self,k_camb,Pk_camb,grid_k,cell_size,L_max):
		self.Pk_camb_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_camb)	     #interpolate camb's Power Spectrum
		##########################################
		# Integral limits for the k-space integral
		##########################################
		k_min = np.min(k_camb[1:])									     
		k_maxbox = (2.*np.pi)/cell_size			# used in the dumping term to eliminate oscillations
		k_max = float(k_camb[-1:])/10
		self.k_max = k_max							     

		k_step = 1./L_max*(1./0.3)				# the step in which the integral will be calculated
		k_r = np.arange(k_min,3.*k_max,k_step)
		###########################
		#	Finding Xi_camb(r)
		###########################
		###########################################
		# Integral limits for the r-space integral
		###########################################
		r_max = (np.pi)/np.min(k_r[1:])*(0.3)					#there's no reason to choose this 0.3
		r_step = 1./np.max(grid_k)*(2./3.)
		r_k=1.0*np.arange(1.,r_max,r_step)

		dk_r=np.diff(k_r)                                      # makes the diff between k and k + dk
		dk_r=np.append(dk_r,[0.0])
		krk = np.outer(k_r,r_k)
		self.krk = krk
		#################################
		# Calculates the sin(kr)/kr term
		#################################
		sinkr=np.sin(krk)/krk
		#############################################
		# calculates dk*k^2*P(k)*exp(...)--> dumping
		#############################################
		dkkPk=dk_r*(k_r**2)*self.Pk_camb_interp(k_r)*np.exp(-1.0*np.power(k_r/(3.*k_maxbox),6.0))
		############################
		# The first integral itself
		############################
		integral=np.einsum('i,ij->j',dkkPk,sinkr)
		#############################
		# Cambs correlation function
		#############################
		self.corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*integral #*np.sum(integrando,axis=0)       		     # uses the trace in the r axis to make the integral
		####################################
		# The gaussian correlation function
		####################################
		corr_g = np.log(1.+self.corr_ln) 
		
		########################
		#	Calculating P_g(k)
		########################
		dr = np.diff(r_k)
		dr = np.append(dr,[0.0])
		rkr = np.outer(r_k,k_r)
		####################################
		# sin(rk)/rk and dr*r^2*Xi_gauss(r)
		####################################
		sinrk2 = np.sin(rkr)/rkr
		drCorr = dr*r_k*r_k*corr_g

		##############################################################
		# The second integral, resulting in a Gaussian Power Spectrum
		##############################################################
		integral2 = np.einsum('j,ji->i',drCorr,sinrk2)
		
		Pk_gauss = 4.*np.pi*integral2  #np.sum(integrando2, axis=0)								
		Pk_gauss[0] = Pk_camb[1]
		##############################################
		# The values that can be returned by the class
		##############################################
		self.Pk_gauss_interp = interpolate.UnivariateSpline(k_r,Pk_gauss)
		self.k_r = k_r
		self.r_k = r_k
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
