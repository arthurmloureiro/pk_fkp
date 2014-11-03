#!usr/bin/env python

from time import clock
import numpy as np

def FKP(gridname,num_bins,n_bar,bias,cell_size,n_x):
    
    phsize=float(cell_size*n_x) #physical size of the side of the (assumed square) grid
    
    ng=gridname #the galaxy field to be analyzed
    n_bar_matrix = np.ones((n_x,n_x,n_x))*n_bar #the 3D version of n_bar
    largenumber = 1000
    nr = np.random.poisson(n_bar_matrix)*largenumber #the random catalog, with the same selection function n_bar, but with a lot more galaxies
    
    ###############################################################
    
    print '\nDefining overdensity field'

    #definitions from Percival, Verde & Peacock 2004 (PVP)
    #the units in this subroutine are NOT physical: volume = (number of cells)^3


    Pi = 5000.0*((n_x/phsize)**3) # initial guess for the power spectrum (arbitrary)

    w = ((bias**2)*Pi) / (1.0+n_bar_matrix*Pi*(bias**2)) #weights according to eq.28 (in PVP)

    alpha = 1.0/largenumber #alpha in PVP

    N = np.sqrt(np.sum((n_bar_matrix**2)*(w**2))) #normalization given by eq.7 (PVP)

    Pshot = ((1+alpha)/(N**2)) * np.sum(n_bar_matrix*((w**2)/(bias**2))) #shot noise, eq. 16 (PVP)
    
    kfft=np.fft.fftfreq(n_x) #FFT frequencies
    kmin=np.amin(np.abs(kfft)) #the minimum frequency; must be 0
    kmaxfft=np.amax(np.abs(kfft)) #finds the Nyquist frequency
    kmax=np.sqrt(3)*kmaxfft #highest possible frequency (yes, it will be greater than the Nyquist frequency)
    krfft=np.abs(kfft[:np.argmax(np.abs(kfft))+1])
    krfft2=krfft**2
    kNy=kmaxfft #Nyquist frequency

    k_bins=np.linspace(kmin,kmax,num_bins-1) #edges of the bins in which P(k) will be estimated
    delta_k=k_bins[4]-k_bins[3]


    F=(w/(N*bias)) * (ng-alpha*nr) #overdensity field, eq. 6 in PVP

    ###############################################################

    print '\nTaking Fourier transform of the overdensity field'

    Fk=np.fft.rfftn(F) #numpy.fft is in the same Fourier convention of PVP - no extra normalization needed
    Fk=Fk
    Fk2=(Fk*Fk.conj()).real #square of the absolute value

    ###############################################################

    
    P_ret=np.zeros(num_bins) #initializing the Power Spectrum that will be the output of the external function
    counts=np.zeros(num_bins) #initializing the vector that averages over modes within a bin 
    init=clock()

    for i in range(len(kfft)):
            	kx2=kfft[i]**2
                for j in range(len(kfft)):
        		ky2=kfft[j]**2
           
        		k_sum = np.sqrt(kx2 + ky2 + krfft2) #absolute value of k
                        m = np.asarray(k_sum/delta_k-0.000001).astype(int)
			#m = np.digitize(k_sum,k_bins) #finds which bin the absolute value is in
         
        		zcounter=0
        		for ind in m: #iterating over the indices to attribute the power to the correct bins
                            #print 'ind=',ind,'zcounter=',zcounter,'Pshape=',P_ret.shape
                            P_ret[ind]=P_ret[ind]+Fk2[i,j,zcounter]
                            counts[ind]=counts[ind]+1
                            zcounter=zcounter+1
          
    fin=clock()
    print '---averaging over shells in k-space took',fin-init,'seconds'

    P_ret = P_ret/counts - Pshot #mean power on each bin and shot noise correction

    ###############################################################

    print '\nCalculating error bars'


    init=clock()
    rel_var=np.zeros(len(P_ret)) #initializing relative variance vector

    nbarw2=(n_bar_matrix*w)**2
    pifactor=((2*np.pi)**3)/(N**4) #useful expressions
    nbarwb2=(n_bar_matrix)*((w/bias)**2)


    for i in range(len(P_ret)):
        rel_var[i]=( (pifactor) * np.sum( (nbarw2 + nbarwb2/P_ret[i])**2 )) #eq. 26 from PVP, except for the V_k term, which I include a few lines ahead


    fin=clock()
    print '---took',fin-init,'seconds'
    
    V_k = counts/ ( (n_x/2.0)*n_x**2) #this factor of volume is the fraction of modes that fell within each bin, makes more sense in this discrete case instead of 4*pi*(k**2)*(delta k)
    rel_var=rel_var/V_k
    sigma=np.sqrt(rel_var*P_ret**2) #1-sigma error bars vector

    ###############################################################

    k=np.zeros(len(P_ret))
    for i in range(len(k_bins)-1): #power in the center of the bin
        k[i]=(k_bins[i]+k_bins[i+1])/2.0
    
    #print k,P_ret,sigma,Pshot
    #changing to physical units
    sigma=sigma*((phsize/n_x)**3)
    P_ret=P_ret*((phsize/n_x)**3) 
    Pshot=Pshot*((phsize/n_x)**3) 
    k=k*(2*np.pi*n_x/phsize)
    #print k,P_ret,sigma,Pshot
   

    #eliminating the first 2 and last value, which are problematic, should be fixed
    P_ret=P_ret[1:]
    k=k[1:]
    sigma=sigma[1:]

    return (k,P_ret,sigma)
