#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Creates a 3D grid in Fourier space that obeys how the FFT behaves in python

	v0.1
	v1.0 - In 3D
	v1.5 - It can plot slices of the matrix
	v1.7 - Uses the side of the box 
	v2.0 - Uses Einsum to generate the grid
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
####################################################
# Uncomment the line above and the last three lines 
# if you have matplotlib and want to see the grid
####################################################
class grid3d:
	'''
	The input is the size of the vectors k_x, k_y and k_z
	'''
	def __init__(self,n_x,n_y,n_z,L_x,L_y,L_z):
		self.size_x = n_x
		self.size_y = n_y
		self.size_z = n_z
		self.Lx = L_x
		self.Ly = L_y
		self.Lz = L_z
		#######################################
		# k0 has to be this value so |k|<k_max
		#######################################
		kx0 = (2*np.pi)/L_x				
		ky0 = (2*np.pi)/L_y
		kz0 = (2*np.pi)/L_z
		
		###################################################################
		# see numpy documentation for np.fft.fftfreq details
		###################################################################
		self.k_x = (n_x)*np.fft.fftfreq(n_x)*kx0
		identx = np.ones_like(self.k_x)

		self.k_y = (n_y)*np.fft.fftfreq(n_y)*ky0
		identy = np.ones_like(self.k_y)	

		self.k_z = (n_z)*np.fft.fftfreq(n_z)*kz0
		identz = np.ones_like(self.k_z)	
		
		self.KX2 = np.einsum('i,j,k', self.k_x*self.k_x,identx,identx)
		self.KY2 = np.einsum('i,j,k', identy,self.k_y*self.k_y,identy)
		self.KZ2 = np.einsum('i,j,k', identz,identz,self.k_z*self.k_z)
		
		self.grid_k = np.sqrt(self.KX2 + self.KY2 + self.KZ2)

		####################################################		
		# Generating a grid a real space, uses grid unities
		####################################################
		r_x = np.arange(n_x)+50#*(L_x/n_x)
		r_y = np.arange(n_y)+50#*(L_y/n_y)
		r_z = np.arange(n_z)+50#*(L_z/n_z)
		self.RX2 = np.einsum('i,j,k', r_x*r_x,identx,identx)
		self.RY2 = np.einsum('i,j,k', identy,r_y*r_y,identy)
		self.RZ2 = np.einsum('i,j,k', identz,identz,r_z*r_z)
		self.grid_r = np.sqrt(self.RX2 + self.RY2 + self.RZ2)


#		pl.figure("Matriz de k")

#		self.plot = pl.imshow(self.matrix[3], cmap=cm.jet)
#		self.plot = pl.imshow(self.grid_r[3], cmap=cm.jet)#, interpolation="nearest")
#		pl.colorbar()
		#self.plothist = pl.imshow(self.hist[3], cmap=cm.jet)
		#pl.show()