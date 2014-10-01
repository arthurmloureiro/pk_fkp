#!usr/bin/env python
# -*- coding: utf-8 -*-

########################################
#    CIC - Cloud-in-Cell Assignment    #
#                                      #
#                                      #
#   -= Lucas Secco, 20/05/14, SP =-    # 
########################################
'''
Código que tem como input um catálogo com colunas (x,y,z) descrevendo a posição de galáxias. O output um grid cúbico de lado 'gridsize'.

É necessário saber qual o lado físico ('phsize') do catálogo original 
'''

import numpy as np
import pylab as pl
from time import clock

################################################
'''definições iniciais'''

gridsize=128 #arbitrário
data='red_gal.dat' #nome do catálogo onde estão os dados
phsize=1000.0 # [Mpc.h^-1], pode variar de catálogo para catálogo


################################################
'''carregando dados'''

print '\nLoading data (%s)'%data

xdat,ydat,zdat,ndat=pl.loadtxt(data,unpack=True,usecols=(0,1,2,3))


nobj=len(xdat) #número de objetos no catálogo

ng=np.zeros((gridsize+1,gridsize+1,gridsize+1)) #inicializando o grid inicial (sem correções de condições de contorno)

ngx=np.linspace(0,phsize,gridsize+1) #gerando limites dos "bins" do grid
ngy=np.linspace(0,phsize,gridsize+1)
ngz=np.linspace(0,phsize,gridsize+1)
dx=ngx[1]-ngx[0] #definindo elemento de volume, a normalização da nuvem do CIC
dy=ngy[1]-ngy[0]
dz=ngz[1]-ngz[0]

################################################
'''calculando o CIC'''

print '\nPerforming CIC'



indx_vec=np.digitize(xdat, ngx)
indy_vec=np.digitize(ydat, ngy)
indz_vec=np.digitize(zdat, ngz) 
#essa parte é importante, essa rotina np.digitize(a,b) distribui os valores em a dentro de bins com fronteiras dadas por b e retorna ***o índice*** do bin referente a cada um dos valores. É basicamente com essa rotina que eu evito loops com condicionais nesse CIC.
#exemplo:
# a = array([ 0.36272769,  0.0508066 ,  0.15092746,  0.34734485,  0.53401666])
# b = array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]) 
# np.digitize(a,b)
# >>> array([2, 1, 1, 2, 3])


init=clock()
for i in range(len(xdat)):

   x=xdat[i]
   indx=indx_vec[i]  #definindo os índices, x, x(i) e x(i-1)
   xi=ngx[indx]
   xi_1=ngx[indx-1]

   y=ydat[i]
   indy=indy_vec[i]
   yi=ngy[indy]
   yi_1=ngy[indy-1]

   z=zdat[i]
   indz=indz_vec[i]
   zi=ngz[indz]
   zi_1=ngz[indz-1]
 

 #continha principal do CIC:
   ng[indx-1,indy-1,indz-1]= ng[indx-1,indy-1,indz-1] + (xi - x) * (yi - y) * (zi - z) / (dx*dy*dz)

   ng[indx,indy-1,indz-1]= ng[indx,indy-1,indz-1] + (x - xi_1) * (yi - y) * (zi - z) / (dx*dy*dz)

   ng[indx,indy,indz-1]= ng[indx,indy,indz-1] + (x - xi_1) * (y - yi_1) * (zi - z) / (dx*dy*dz)

   ng[indx-1,indy,indz-1]= ng[indx-1,indy,indz-1] + (xi - x) * (y - yi_1) * (zi - z) / (dx*dy*dz)

   ng[indx-1,indy,indz]= ng[indx-1,indy,indz] + (xi - x) * (y - yi_1) * (z - zi_1) / (dx*dy*dz)

   ng[indx,indy-1,indz]= ng[indx,indy-1,indz] + (x - xi_1) * (yi - y) * (z - zi_1) / (dx*dy*dz) 

   ng[indx-1,indy-1,indz]= ng[indx-1,indy-1,indz] + (xi - x) * (yi - y) * (z - zi_1) / (dx*dy*dz)

   ng[indx,indy,indz]= ng[indx,indy,indz] + (x - xi_1) * (y - yi_1) * (z - zi_1) / (dx*dy*dz)


fin=clock()
print '--o loop levou',fin-init,'segundos'

print '--checking sum:',np.sum(ng),'out of',len(xdat)
################################################
'''incluindo condições de contorno'''

print "\nAdjusting periodic boundary conditions"
# Agora defino uma nova matriz, que vai ser o output do código. Da forma que o CIC está sendo feito, os bins nas bordas do grid serão sempre muito menos preenchidos do que deveriam. Isso acontece porque o código não assume sozinho condições de contorno periódicas pra contribuição de cada galáxia na densidade do grid. Essa nova ng_final é justamente para inserir essa periodicidade: os 8 pontos nas bordas são somados e os lados extremamente opostos também.

ng_final=np.zeros((gridsize,gridsize,gridsize))

ng_final[0,0,0] = ng[0,0,0]+ng[0,-1,-1]+ng[0,-1,0]+ng[0,0,-1]+ng[-1,0,0]+ng[-1,-1,-1]+ng[-1,-1,0]+ng[-1,0,-1] #8 cantos

for i in range(1,len(ng_final)): #12 arestas
   ng_final[i,0,0]=ng[i,0,0]+ng[i,-1,0]+ng[i,-1,-1]+ng[i,0,-1]
   ng_final[0,i,0]=ng[0,i,0]+ng[-1,i,0]+ng[-1,i,-1]+ng[0,i,-1]
   ng_final[0,0,i]=ng[0,0,i]+ng[-1,0,i]+ng[-1,-1,i]+ng[0,-1,i]

for i in range(1,len(ng_final)): 
    for j in range(1,len(ng_final)): #6 faces
        ng_final[i,j,0]=ng[i,j,0]+ng[i,j,-1]
        ng_final[0,i,j]=ng[0,i,j]+ng[-1,i,j]
        ng_final[j,0,i]=ng[j,0,i]+ng[j,-1,i]

for i in range(1,len(ng_final)): #1 bulk
    for j in range(1,len(ng_final)):
        for k in range(1,len(ng_final)):
            ng_final[i,j,k]=ng[i,j,k]


print 'checking sum:',np.sum(ng_final),'out of',np.sum(ng) #pra verificar que nenhuma galáxia está sendo perdida ou criada, um teste é ver se a soma de todos os elementos de ng_final dá a mesma coisa que a soma de todos os elementos de ng!

#############################################
''' para visualizar o resultado (uma fatia do grid) '''

bla=np.random.randint(0,gridsize) #gera um inteiro aleatório tirado de range(gridsize) 

#pl.figure('ng')
#pl.imshow(ng[bla],interpolation='bilinear')
#pl.colorbar()

pl.figure('ng_final')
pl.imshow(ng_final[bla],cmap='afmhot',interpolation='gaussian')
pl.colorbar()
pl.title('fatia %d'%bla)



##############################################
''' escrevendo o arquivo de texto com o grid '''

outname=data[0:-4]+'%d_CICgrid.dat'%gridsize

print '\nWriting grid in file %s'%outname

output=open(outname,'w')

output.write('#grid with %d cells from %s\n#use .reshape(%d,%d,%d)!!\n' % (gridsize,data,gridsize,gridsize,gridsize))

for i in range(len(ng_final)):
   for j in range(len(ng_final)):
      for k in range(len(ng_final)):
         output.write("%f\n" % ng_final[i,j,k])
output.close()

###############################################
print "\nDone!"

pl.show() #comentar aqui para a visualização do grid não ser gerada


