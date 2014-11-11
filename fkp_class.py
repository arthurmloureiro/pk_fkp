#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
"""
from time import clock
import numpy as np
    
class fkp_init(object):
	'''
	This class contains the initial calculations for the FKP routine
	It still needs testing
	but this is the part to be called outside the N_realz loop, since w and N are
	the same for all realizations
	Enters num_bins, n_bar (a matrix), bias 
	and the number of cells of the cubic map    
	'''
	def __init__(self,num_bins,n_bar_matrix,bias,cell_size,n_x,bin_matrix):    
		self.num_bins = num_bins
		self.n_bar_matrix = n_bar_matrix
		self.bias = bias
		self.cell_size = cell_size
		self.n_x = n_x
		self.bin_matrix = bin_matrix
		self.phsize=float(cell_size*n_x)	#physical size of the side of the (assumed square) grid
        #n_bar_matrix = np.ones((n_x,n_x,n_x))*n_bar 	#the 3D version of n_bar
                                # this needs change!!!!!!
		largenumber = 1000
#		largenumber = 100
		self.nr = np.random.poisson(n_bar_matrix)*largenumber #the random catalog, with the same selection function n_bar, but with a lot more galaxies
		###############################################################
		#print '\nDefining overdensity field'

		#definitions from Percival, Verde & Peacock 2004 (PVP)
		#the units in this subroutine are NOT physical: volume = (number of cells)^3

		self.Pi = 5000.0*((n_x/self.phsize)**3) # initial guess for the power spectrum (arbitrary)
		self.w = ((bias**2)*self.Pi) / (1.0+n_bar_matrix*self.Pi*(bias**2)) #weights according to eq.28 (in PVP)
		self.alpha = 1.0/largenumber #alpha in PVP
		self.N = np.sqrt(np.sum((n_bar_matrix**2)*(self.w**2))) #normalization given by eq.7 (PVP)
		self.N2 = np.sum((n_bar_matrix**4.)*(self.w**4.))
		self.N3 = np.sum((n_bar_matrix**3.)*(self.w**4.)/(bias**2.))
		self.N4 = np.sum((n_bar_matrix**2.)*(self.w**4.)/(bias**4.))
		self.Pshot = ((1+self.alpha)/(self.N**2)) * np.sum(n_bar_matrix*((self.w**2)/(bias**2))) #shot noise, eq. 16 (PVP)kfft=np.fft.fftfreq(n_x) #FFT frequencies
		
		#self.kfft=np.fft.fftfreq(n_x) #FFT frequencies
		#kmin=np.amin(np.abs(self.kfft)) #the minimum frequency; must be 0
		#kmaxfft=np.amax(np.abs(self.kfft)) #finds the Nyquist frequency
		#kmax=np.sqrt(2.7)*kmaxfft #highest possible frequency (yes, it will be greater than the Nyquist frequency)
		#krfft=np.abs(self.kfft[:np.argmax(np.abs(self.kfft))+1])
		#self.krfft2=krfft**2
		#kNy=kmaxfft #Nyquist frequency
		#self.kNy = kNy
		#self.k_bins=np.linspace(kmin,kmax,self.num_bins-1)
		#self.delta_k=self.k_bins[4]-self.k_bins[3]
    
	def fkp(self,ng):
		'''
		This is the FKP function itself, only entry is the galaxy map
		'''
		self.F=(self.w/(self.N*self.bias)) * (ng-self.alpha*self.nr) #overdensity field, eq. 6 in PVP
		###############################################################
		
		#print '\nTaking Fourier transform of the overdensity field'

		Fk=np.fft.rfftn(self.F) #numpy.fft is in the same Fourier convention of PVP - no extra normalization needed
        #Fk=np.fft.fftn(self.F) #numpy.fft is in the same Fourier convention of PVP - no extra
		Fk=Fk                   #?????????
		#Fk2=(Fk*Fk.conj()).real #square of the absolute value

        ###############################################################
		#P_ret=np.zeros(self.num_bins) #initializing the Power Spectrum that will be the output of the external function
		counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin 
		init=clock()
		#print Fk2
		#for i in range(len(self.kfft)):
		#	kx2=self.kfft[i]**2
		#	for j in range(len(self.kfft)):
		#		ky2=self.kfft[j]**2
	
		#		k_sum = np.sqrt(kx2 + ky2 + self.krfft2) #absolute value of k
		#		m = np.asarray(k_sum/self.delta_k-0.000001).astype(int)
                #m = np.asarray(k_sum/self.delta_k-0.5).astype(int)
           #m = np.digitize(k_sum,k_bins) #finds which bin the absolute value is in         
		#		zcounter=0
		#		for ind in m: #iterating over the indices to attribute the power to the correct bins
                    #print 'ind=',ind,'zcounter=',zcounter,'Pshape=',P_ret.shape
		#			P_ret[ind]=P_ret[ind]+Fk2[i,j,zcounter]
		#			counts[ind]=counts[ind]+1
		#			zcounter=zcounter+1
		self.counts2 = np.einsum("aijl->a", self.bin_matrix)
		counts = np.einsum("aijl->a", self.bin_matrix)
		self.counts = counts
		P_ret = np.einsum("aijl,ijl,ijl->a", self.bin_matrix, Fk, np.conj(Fk))/np.einsum("aijl->a", self.bin_matrix)
		fin=clock()
		#print '---averaging over shells in k-space took',fin-init,'seconds'

		#P_ret = P_ret/counts - self.Pshot #mean power on each bin and shot noise correction
		P_ret = P_ret - self.Pshot
        ###############################################################

		#print '\nCalculating error bars'

		init=clock()
		rel_var2=np.zeros(len(P_ret)) #initializing relative variance vector

		#nbarw2=(self.n_bar_matrix*self.w)**2
		pifactor=((2*np.pi)**3)/(self.N**4) #useful expressions
		#nbarwb2=(self.n_bar_matrix)*((self.w/self.bias)**2)
		#for i in range(len(P_ret)):
		#	rel_var2[i]=( (pifactor) * np.sum( (nbarw2 + nbarwb2/P_ret[i])**2 )) #eq. 26 from PVP, except for the V_k term, which I include a few lines ahead
#		rel_var = pifactor*(self.N2 + 2.*self.N3*np.power(P_ret,-1.) + self.N4*np.power(P_ret,-2.))
		rel_var = pifactor*(self.N2 + 2.*self.N3/P_ret + self.N4/(P_ret**2))
		#self.rel_var = rel_var
		#self.rel_var2 = rel_var2
		fin=clock()
		#print '---took',fin-init,'seconds'

		V_k = counts/ ( (self.n_x/2.0)*self.n_x**2) #this factor of volume is the fraction of modes that fell within each bin, makes more sense in this discrete case instead of 4*pi*(k**2)*(delta k)
		rel_var=rel_var/V_k
		sigma=np.sqrt(rel_var*P_ret**2) #1-sigma error bars vector
		
        ###############################################################
		#k=np.zeros(len(P_ret))
		#for i in range(len(self.k_bins)-1): #power in the center of the bin
	#		k[i]=(self.k_bins[i]+self.k_bins[i+1])/2.0

		#changing to physical units

		P_ret=P_ret*((self.phsize/self.n_x)**3) 
		#P_ret2 = P_ret2*((self.phsize/self.n_x)**3)
		self.Pshot_phys=self.Pshot*((self.phsize/self.n_x)**3) 
		#k=k*(2*np.pi*self.n_x/self.phsize)
		sigma=sigma*((self.phsize/self.n_x)**3)
        #eliminating the first 2 and last value, which are problematic, should be fixed
		self.P_ret=P_ret
		#self.P_ret2=P_ret2[1:]
		#self.kk=k
		self.sigma=sigma
		self.Fk = Fk
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
