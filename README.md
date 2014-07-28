pk_fkp
======
= 
Table of content
---
- **grid3D.py** - class that generates the 3D-grid in Fourier Space in a way that respects python's FFT algorithm, also a grid in real space
- **pk_fkp.py** - main body of the program
- **input.dat** - the input file for the main program

Recent Modifications:
---
- 18/07/2014 - Repository Created, *grid3D.py* cloned, *input file* crated, implemented part of the *pk.py* code in the main code [A]
- 21/07/2014 - Main code ready and waiting for the implementation of the fkp part [A]
- 28/07/2014 - grid3d.py now uses np.fft.fftfreq to generate the k-grid an generates a r-grid also. Main code modified to contain the selection function n_bar
