#!usr/bin/env python
# -*- coding: utf-8 -*-

###########################################
#              estimador FKP              # 
#                  (v1.1)                 #
#                                         #
#*para apenas 1 grid                      #
#*pode incluir efeitos de seleção         #
#                                         #
#   -= Lucas Secco, Junho 2014, SP =-     #
###########################################


import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as sco
from time import clock

####################################################################
'''definições iniciais'''

#os inputs abaixo são informação prévia sobre o grid 

gridsize=100 #o número de células
bins= 30 #número de bins em k onde P vai ser estimado
catalog= 'catalog_lowdensity.dat' #nome do arquivo a ser usado como catálogo
#random_catalog= 'random_'+catalog #nome do catálogo aleatório
spectrum_output= 'P_'+catalog #nome do arquivo de saída
phsize= 500.0 #[h^-1.Mpc] tamanho físico do lado do grid (cúbico)
b=1.0 #chute inicial para o bias do traçador 


###################################################################
'''lendo os dados'''

print '\nLoading data (%s)'%catalog

dat=np.loadtxt(catalog,unpack=True,comments='#',usecols=(3,)) #lê os dados como um array 1D


###################################################################
'''construindo os grids'''

print '\nDefining grids'

ng=dat.reshape(gridsize,gridsize,gridsize) #transforma o grid que antes era uma sequência 1D de números em um grid cúbico


galaxynumber=np.sum(ng) #número total de galáxias no grid
ratio=1000.0 # 1/alpha, que aparece na equação 6 de PVP
mean_real=galaxynumber/((gridsize)**3) #média de galáxias
mean_random=ratio*mean_real


#################################
''' construindo nbar '''

#Aqui é criado um grid chamado nbar que vai ser basicamente a função de seleção, a função "fit(r)" abaixo é a função de seleção e o loop abaixo preenche o grid com os valores da função em cada ponto

'''
print '--defining nbar'

def fit(r):
    n0=0.000543
    factor1=0.00051565*r
    factor2=(r**2)*1.4366e-06
    factor3=(r**3)*7.088e-10
    factor4=(r**4)*1.31e-14
    e=n0*np.exp(-factor1+factor2-factor3+factor4)*(phsize/gridsize)**3
    return e

nbar=np.zeros((gridsize,gridsize,gridsize))
for i in range(gridsize):
    for j in range(gridsize):
        for l in range(gridsize):
            #r=(20.0/4.434)*np.sqrt(i**2 + j**2 +l**2) #pq eu dividi por esse número 4.434??
            r=20.0*np.sqrt((i+50)**2 + (j+50)**2 + (l+50)**2)
            nbar[i,j,l]=fit(r)

'''
nbar=np.ones((gridsize,gridsize,gridsize))*mean_real #se a parte de cima estiver comentada, esse pedaço gera um grid com uma seleção constante e igual à média do número de galáxias do grid


#################################
#para gerar o catálogo random
print '---generating random catalog'
nr=np.zeros((gridsize,gridsize,gridsize))
for i in range(gridsize):
    for j in range(gridsize):
        for l in range(gridsize):
            nr[i,j,l]=np.random.poisson(nbar[i,j,l]*ratio) #um processo de poisson em cada ponto


##################################
#para ler um catálogo random já existente (que esteja no formato de uma sequência 1D)
'''
print '---loading random catalog'
rand=np.loadtxt(random_catalog,delimiter=',')
rand=rand.reshape(gridsize,gridsize,gridsize)
'''
################################### 
#para eliminar fileiras do grid resultante, caso necessário  
'''
print '---eliminating last row/column/depth' #o loop lê só o 'bulk'
ng=np.zeros((gridsize,gridsize,gridsize))
nr=np.zeros((gridsize,gridsize,gridsize))
for i in range(gridsize):
    for j in range(gridsize):
        for k in range(gridsize):
            ng[i,j,k]=ing[i,j,k]
            nr[i,j,k]=inr[i,j,k]
'''


#####################################################################
'''definindo o campo de sobredensidades, em etapas...'''

print '\nDefining overdensity field F'

#todas as definições utilizadas são do artigo de Percival, Verde & Peacock 2004
#cuidado a partir de agora: eu estou definindo tudo em unidades DO GRID (ou seja, volume = gridsize**3 adimensional)


Pi=5000.0*((gridsize/phsize)**3) # chute inicial pro espectro, talvez possa ser constante mesmo

w=((b**2)*Pi) / (1.0+nbar*Pi*(b**2)) #pesos segundo eq.28 de PVP

alpha=1.0/ratio #ratio está definido lá no começo

N=np.sqrt(np.sum((nbar**2)*(w**2))) #normalização dada pela eq.7 de PVP

#as partes abaixo plotam uma fatia de nbar (a função de seleção), ng (o grid), nr (o grid random) e w (os pesos). serve como um "sanity check"

'''
pl.figure('nbar')
pl.imshow(nbar[int(gridsize/2)],cmap='afmhot')
pl.colorbar()

pl.figure('ng')
pl.imshow(ng[int(gridsize/2)],cmap='afmhot')
pl.colorbar()

pl.figure('nr')
pl.imshow(nr[int(gridsize/2)],cmap='afmhot')
pl.colorbar()

pl.figure('w')
pl.imshow(w[int(gridsize/2)],cmap='afmhot')
pl.colorbar()

pl.show()
'''

############################
''' definindo shot noise '''

Pshot=((1+alpha)/(N**2)) * np.sum(nbar*((w**2)/(b**2))) #eq. 16 de PVP

############################
'''calculando o espectro'''
print '\nEstimating P(k)'


kfft=np.fft.fftfreq(gridsize) #gera exatamente as frequências correspondentes à fft
kminfft=np.amin(np.abs(kfft)) #encontra a menor frequência (certamente deve ser 0)
kmaxfft=np.amax(np.abs(kfft)) #encontra a maior frequência (Nyquist), que pode ser negativa
kmax=np.sqrt(3)*kmaxfft #o maior módulo de k possível é dado por isso
kmin=kminfft #o menor módulo de k possível é dado por isso (certamente deve ser 0)

kNy=kmaxfft #freq. de Nyquist

k_bins=np.linspace(kmin,kmax,bins+1) #selecionando as fronteiras dos bins onde k vai ser estimado


F=(w/(N*b)) * (ng-alpha*nr) #campo de sobredensidades, eq. 6 de PVP

#######################################
'''tomando a transformada de Fourier'''


Fk=np.fft.fftn(F) #a tranformada de Fourier está na mesma convenção de PVP e não é necessário nenhum fator extra de normalização

Fk2=(Fk*Fk.conj()).real #módulo quadrado

########################################
'''fazendo médias sobre cascas no espaço recíproco'''

#ISSO PRECISA SER OTIMIZADO!!! é a parte mais lenta do código porque eu não estou calculando as médias usando matrizes


P=np.zeros(bins) #inicializando vetor do espectro final
counts=np.zeros(bins) #incializando vetor das contagens de modos

init=clock()
for i in range(len(kfft)):
    for j in range(len(kfft)): #esses 3 loops cobrem todas as combinações possíveis de coordenadas no espaço recíproco
        for l in range(len(kfft)):
            
            kx=kfft[i]
            ky=kfft[j]
            kz=kfft[l]
            k_sum=np.sqrt(kx**2 + ky**2 + kz**2) #módulo da distância até a origem no espaço recíproco, afinal quero estimar P(k) e não P(\vec{k})
            
            for m in range(len(k_bins)-1):
                if (k_sum>=k_bins[m] and k_sum<=k_bins[m+1]): #se estiver dentro do bin, adiciona a potência a esse bin
                    P[m]=P[m]+Fk2[i,j,l]-Pshot
                    counts[m]=counts[m]+1
                    break
fin=clock()
print '--averaging over shells in k-space took',fin-init,'seconds'

P=P/counts #faz a média sobre o número de modos

   

#####################################################################
''' calculando barras de erro '''

print '\nCalculating error bars'


rel_var=np.zeros(len(P)) #inicializando vetor da variância relativa

for i in range(len(P)):
    rel_var[i]=( ((2*np.pi)**3)/(N**4)) * np.sum( ((nbar*w)**2 + (nbar)*((w/b)**2)/P[i])**2 ) #eq. 26 de PVP, exceto pelo termo V_k, que eu estou incluindo algumas linhas abaixo

rel_var=rel_var/(counts/(gridsize**3))

sigma=np.sqrt(rel_var*P**2) #agora sim, esse é o vetor com as barras de erro de 1-sigma

#####################################################################
''' plotando espectro '''

print "\nPlotting (convolved) power spectrum"


k_camb,P_camb=np.loadtxt('Pk-input.dat',unpack=True)

k_nlin,P_nlin=np.loadtxt('nlinLambda_matterpower.dat',unpack=True)
 

k=np.zeros(len(P))
for i in range(len(k_bins)-1): #só pra associar cada potência ao centro do bin
    k[i]=(k_bins[i]+k_bins[i+1])/2.0




#as linhas abaixo convertem as unidades do código de GRID pra REAL (ou seja, uma conversão entre unidades adimensionais e Mpc.h^-1)
sigma=sigma*((phsize/gridsize)**3)
P=P*((phsize/gridsize)**3) #dando ao espectro unidades de [h^-3.Mpc^3]
Pshot=Pshot*((phsize/gridsize)**3) 
k=k*(2*np.pi*gridsize/phsize) #dando a k unidades de [h.Mpc^-1]
kNy=kNy*(2*np.pi*gridsize/phsize)

#as linhas abaixo eliminam o primeiro valor estimado do espectro, que é problemático por causa da definião de alpha (explicação em PVP)    
P=P[1:] 
k=k[1:]
sigma=sigma[1:]

pl.figure('P(k)')
pl.plot(k_camb,P_camb,'b-',label='camb (linear)')
pl.plot(k_nlin,P_nlin,'g-',label='camb (nonlinear)')
pl.errorbar(k,P,yerr=sigma,fmt='ro',label='estimated P(k)')
pl.axvline(x=kNy,color='y',label='Nyquist frequency')
pl.title(r'P(k) for %s in %d bins with box size %d, Pshot=%f '%(catalog,bins,gridsize,Pshot))
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=19)
pl.ylabel(r'$P(k)\,[h^{-3}$Mpc$^{3}]$',fontsize=19)
pl.grid(True)
pl.legend()
pl.loglog()


#####################################################################
''' criando .dat com a saída do código (k,P,erro) '''


print '\nWriting estimated P(k) with errorbars in %s'%spectrum_output
out=open(spectrum_output,'w')
out.write('#k[h.Mpc^-1] P[h^-3.Mpc^3] 1sigma[h^-3.Mpc^3]')
for i in range(len(k)):
    out.write('%f %f %f\n'%(k[i],P[i],sigma[i]))
out.close()    


print "\nDone!\n"

pl.show()


