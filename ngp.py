#!usr/bin/env python
# -*- coding: utf-8 -*-

########################################
# NGP - Nearest Grid Point Assignment  #
#                                      #
#*não considera a massa das galáxias   #
#                                      #
#   -= Lucas Secco, 14/08/14, SP =-    # 
########################################
'''
código que tem como input um catálogo com colunas (x,y,z) descrevendo a posição de galáxias numa simulação de N corpos e como output um grid cúbico de lado gridsize.

é necessário saber qual o lado físico (phsize) do grid 
'''

import numpy as np
import matplotlib.pyplot as pl
from time import clock

################################################
'''definições iniciais'''
gridsize=100 #arbitrário
data='aD648L.dat'
phsize=648.0 # [Mpc.h^-1] ,específico pra esse catálogo (pode variar)

################################################
'''carregando dados'''

print '\nLoading data (%s)'%data

xdat,ydat,zdat=np.loadtxt(data,unpack=True,usecols=(0,1,2))


nobj=len(xdat) #número de objetos no catálogo


###############################################
'''calculando o NGP'''

print '\nPerforming NGP'

ini=clock()
x=xdat/phsize # os valores vão de 0 a 1
x=x*(gridsize-1) # os valores vão de 0 a gridsize-1
x=np.round(x) #os valores são arredondados e estão entre 0 e gridsize-1, o que corresponde a um total de gridsize índices no grid (e por isso a multiplicação anterior foi foi gridsize-1)

y=np.round((ydat/phsize)*(gridsize-1))
z=np.round((zdat/phsize)*(gridsize-1))

ng=np.zeros((gridsize,gridsize,gridsize))

for i in range(len(x)):
   ng[x[i],y[i],z[i]] = ng[x[i],y[i],z[i]] +1 #preenchendo o grid com o número de galáxias correto para cada célula, esse é o grid final já
fin=clock()

print '---took',fin-ini,'seconds'

nfin=np.sum(ng)

print '---checking conservation of number: initial=',nobj,'final=',nfin

################################################
''' para visualizar o resultado (uma fatia do grid) '''

bla=np.random.randint(0,gridsize) #gera um inteiro aleatório tirado de range(gridsize) 

pl.figure('ng')
pl.imshow(ng[bla],cmap='afmhot',interpolation='gaussian')
pl.title('fatia %d'%bla)
pl.colorbar()

################################################
''' escrevendo o arquivo de texto com o grid '''

outname=data[0:-4]+'%d_NGPgrid.dat'%gridsize

print '\nWriting output (%s)'%outname

output=open(outname,'w')

output.write('#grid with %d cells from %s\n#use .reshape(%d,%d,%d)!!\n' % (gridsize,data,gridsize,gridsize,gridsize))

for i in range(len(ng)):
   for j in range(len(ng)):
      for k in range(len(ng)):
         output.write("%f\n" % ng[i,j,k])
output.close()

###############################################
print "\nDone!"

pl.show()





