import random
import numpy as np
import copy

class Neurones:
	
	def __init__( self , tailles, fct_activation, drv_activation):
		
		ainit=[[random.random() for k in range(tailles[i])]for i in range(len(tailles))]
		self.a=np.array(ainit) 											#a[i][j] valeur du neurone j de la couche i ,
																		#ce n'est pas une matrice , on initialise aléatoirement
		
		winit=[]
		for l in range(len(tailles)-1):
			winit.append(np.random.rand(tailles[l],tailles[l+1])) 		# np.random.rand(i,j) renvoie une matrice de nombres aléatoires de taille i*j
		
		self.w=np.array([]) #w[L][i,j] est le poids de la connexion entre le neurone i de la couche L et le neurone j de la couche L+1
		
		self.sig=fct_activation		#on appelle sig comme sigma la fonction d'activation du réseau
		self.sigp=drv_activation	#dérivée fournie par l'utilisateur
		
	def propagation(entree):
		self.a[0]=copy.deepcopy(entree)
		for l in range(1,len(tailles)):
			self.a[l]=np.dot(self.a[l-1],self.w[l]) #éventuellement introduire une matrice de biais ici
			
