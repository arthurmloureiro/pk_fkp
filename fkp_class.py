#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Important Modifications by prof. Abramo for a new M matrix 
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
"""
from time import clock
import numpy as np
from scipy.sparse import csc_matrix
    
class fkp_init(object):
    '''
    This class contains the initial calculations for the FKP routine
    but this is the part to be called outside the N_realz loop, since w and N are
    the same for all realizations
    Enters num_bins, n_bar (matrix), bias 
    n_x,n_y, n_z and the bin_matrix
    '''
    def __init__(self,num_bins,n_bar_matrix,bias,cell_size,n_x,n_y,n_z,bin_matrix):    
        # Here bin_matrix is the M-matrix
        self.num_bins = num_bins
        self.n_bar_matrix = n_bar_matrix
        self.bias = bias
        self.cell_size = cell_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.bin_matrix = bin_matrix
        self.phsize_x=float(cell_size*n_x)	#physical size of the side of the (assumed square) grid
        self.phsize_y=float(cell_size*n_y)
        self.phsize_z=float(cell_size*n_z)
        largenumber = 1000

        self.nr = np.random.poisson(n_bar_matrix*largenumber) #the random catalog, with the same selection function n_bar, but with a lot more galaxies
        ###############################################################
        #print '\nDefining overdensity field'

        #definitions from Percival, Verde & Peacock 2004 (PVP)
        #the units in this subroutine are NOT physical: volume = (number of cells)^3
        small = 1e-25
        #self.Pi = 5000.0*((n_x/self.phsize)**3) # initial guess for the power spectrum (arbitrary)
        self.Pi = 5000.0*(n_x/self.phsize_x)*(n_y/self.phsize_y)*(n_z/self.phsize_z)
        self.w = ((bias**2)*self.Pi) / (1.0+n_bar_matrix*self.Pi*(bias**2)) #weights according to eq.28 (in PVP)
        self.alpha = 1.0/largenumber #alpha in PVP
        self.N = np.sqrt(np.sum((n_bar_matrix**2)*(self.w**2))) #normalization given by eq.7 (PVP)
        self.N2 = np.sum((n_bar_matrix**4.)*(self.w**4.))
        self.N3 = np.sum((n_bar_matrix**3.)*(self.w**4.)/(bias**2.))
        self.N4 = np.sum((n_bar_matrix**2.)*(self.w**4.)/(bias**4.))
        self.Pshot = ((1+self.alpha)/(self.N**2 + small)) * np.sum(n_bar_matrix*((self.w**2)/(bias**2 + small))) #shot noise, eq. 16 (PVP)kfft=np.fft.fftfreq(n_x) #FFT frequencies

    
    def fkp(self,ng):
        '''
        This is the FKP function itself, only entry is the galaxy map
        '''
        small = 1e-25
        self.F=(self.w/(self.N*self.bias +small)) * (ng-self.alpha*self.nr) #overdensity field, eq. 6 in PVP
        ###############################################################

        #print '\nTaking Fourier transform of the overdensity field'

        Fk=np.fft.rfftn(self.F) #numpy.fft is in the same Fourier convention of PVP - no extra normalization needed
        #Fk=np.fft.fftn(self.F) #numpy.fft is in the same Fourier convention of PVP - no extra
        Fk=Fk                   #?????????
        Fkflat = np.ndarray.flatten(Fk[:,:,:self.n_z/2+1])  #Raul
        lenkf = len(Fkflat)                                 #Raul
        Fkf2=(Fkflat*(Fkflat.conj())).real #square of the absolute value - Raul

        ###############################################################
        #P_ret=np.zeros(self.num_bins) #initializing the Power Spectrum that will be the output of the external function
        counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin 
        init=clock()
        
        # Here are the operations involving the M-matrix = bin_matrix) - Raul
        #self.counts2 = np.einsum("aijl->a", self.bin_matrix)
        #counts = np.einsum("aijl->a", self.bin_matrix)  # number of points in each bin a
        counts = (self.bin_matrix).dot(np.ones(lenkf))
        self.counts = counts
        
        # This is <|F(k)|^2> on bins [a] - Raul
        P_ret = ((self.bin_matrix).dot(Fkf2))/(self.counts)
        # P_ret = np.einsum("aijl,ijl,ijl->a", self.bin_matrix, Fk, np.conj(Fk))/(np.einsum("aijl->a", self.bin_matrix) + small)

        fin=clock()
        #print '---averaging over shells in k-space took',fin-init,'seconds'

        #P_ret = P_ret/counts - self.Pshot #mean power on each bin and shot noise correction
        P_ret = P_ret - self.Pshot
        for i in range(len(P_ret)):
        	if np.sign(P_ret[i])==-1:
        		P_ret[i]=0.0
        #P_ret[ np.where(P_ret < 0.0) ] = 0.0
        ###############################################################

        #print '\nCalculating error bars'

        init=clock()
        rel_var2=np.zeros(len(P_ret)) #initializing relative variance vector

        #nbarw2=(self.n_bar_matrix*self.w)**2
        #pifactor=((2*np.pi)**3)/(self.N**4 +small) #useful expressions
        pifactor=1./(self.N**4 +small)
        #nbarwb2=(self.n_bar_matrix)*((self.w/self.bias)**2)
        #for i in range(len(P_ret)):
        #	rel_var2[i]=( (pifactor) * np.sum( (nbarw2 + nbarwb2/P_ret[i])**2 )) #eq. 26 from PVP, except for the V_k term, which I include a few lines ahead
        #		rel_var = pifactor*(self.N2 + 2.*self.N3*np.power(P_ret,-1.) + self.N4*np.power(P_ret,-2.))
        #rel_var = pifactor*(self.N2 + 2.*self.N3/(P_ret+small) + self.N4/(P_ret**2 + small))
        rel_var = pifactor*(self.N2*(P_ret**2) + 2.*self.N3*P_ret + self.N4)
        #self.rel_var = rel_var
        #self.rel_var2 = rel_var2
        fin=clock()
        #print '---took',fin-init,'seconds'

        #V_k = counts/ ( (self.n_x/2.0)*self.n_x**2 + small) #this factor of volume is the fraction of modes that fell within each bin, makes more sense in this discrete case instead of 4*pi*(k**2)*(delta k)
        V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.)+small) 
        rel_var=rel_var/(V_k + small)
        sigma=np.sqrt(rel_var).real #1-sigma error bars vector

        #changing to physical units
        #P_ret=P_ret*((self.phsize/self.n_x)**3) 
        P_ret=P_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        #P_ret2 = P_ret2*((self.phsize/self.n_x)**3)
        #self.Pshot_phys=self.Pshot*((self.phsize/self.n_x)**3) 
        self.Pshot_phys=self.Pshot*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        #k=k*(2*np.pi*self.n_x/self.phsize)
        #sigma=sigma*((self.phsize/self.n_x)**3)
        sigma=sigma*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        #eliminating the first 2 and last value, which are problematic, should be fixed
        self.P_ret=np.abs(P_ret)
        #self.P_ret2=P_ret2[1:]
        #self.kk=k
        self.sigma=sigma
        self.Fk = Fk
		