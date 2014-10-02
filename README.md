pk_fkp
======
= 
Table of content
---
- **grid3D.py** - class that generates the 3D-grid in Fourier Space in a way that respects python's FFT algorithm, also a grid in real space
- **pk_fkp.py** - main body of the program
- **input.dat** - the input file for the main program
- **fkp.py** - standard FKP estimator 
- **cic.py** - cloud-in-cell mass assignment scheme
- **ngp.py** - nearest-grid-point mass assignment scheme

Recent Modifications:
---
- 18/07/2014 - Repository Created, *grid3D.py* cloned, *input file* crated, implemented part of the *pk.py* code in the main code [A]
- 21/07/2014 - Main code ready and waiting for the implementation of the fkp part [A]
- 28/07/2014 - grid3d.py now uses np.fft.fftfreq to generate the k-grid an generates a r-grid also. Main code modified to contain the selection function n_bar [A]
- 29/07/2014 - Selection function working and bin-grid also working [A]
- 08/09/2014 - Branch "multi-tracer" modification to use more than 1 bias, estimate the galaxy power spectrum as in the pk.py program. Basically the pk.py code for multi-tracers [A] 
- 02/10/2014 - Correcting the integral for the Corr Func and the Gaussian P(k). Program modified, waiting for the implementation of the *FKP method*. [A]