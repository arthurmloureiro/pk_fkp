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
#from matplotlib import cm
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
		# it has to be up to m/2 + 1 because of the structure of np.arange
		###################################################################
		prime_x=np.arange(1,(n_x/2+1),1)*kx0		
		invert_prime_x = -prime_x[::-1]			
		prime_x = np.insert(prime_x, 0,0)		
		self.k_x = np.append(prime_x,invert_prime_x)		
		ident = np.ones_like(self.k_x)


		prime_y=np.arange(1,(n_y/2+1),1)*ky0		
		invert_prime_y = -prime_y[::-1]			
		prime_y = np.insert(prime_y, 0,0)		
		self.k_y = np.append(prime_y,invert_prime_y)		


		prime_z=np.arange(1,(n_z/2+1),1)*kz0		
		invert_prime_z = -prime_z[::-1]			
		prime_z = np.insert(prime_z, 0,0)		
		self.k_z = np.append(prime_z,invert_prime_z)	
		
		self.KX2 = np.einsum('i,j,k', self.k_x*self.k_x,ident,ident)
		self.KY2 = np.einsum('i,j,k', ident,self.k_y*self.k_y,ident)
		self.KZ2 = np.einsum('i,j,k', ident,ident,self.k_z*self.k_z)
		
		
		self.matrix = np.sqrt(self.KX2 + self.KY2 + self.KZ2)

#		pl.figure("Matriz de k")
#		self.plot = pl.imshow(self.matrix[3], cmap=cm.jet)
		#self.plothist = pl.imshow(self.hist[3], cmap=cm.jet)
		#pl.show()
