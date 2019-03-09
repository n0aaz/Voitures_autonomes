import random
import numpy as np
import copy
import imageio
import matplotlib.pyplot as plt


dt=1/5

circuit=np.zeros((1000,1000))
circuit[55][50]=1
for k in range(1000):
	xx,yy=random.randrange(0,1000),random.randrange(0,1000)
	circuit[xx][yy]=1
	plt.plot(xx,yy)

class Neurones:
	
	def __init__( self , tailles, fct_activation, drv_activation):
		
		ainit=[[random.random() for k in range(tailles[i])]for i in range(len(tailles))]
		self.a=np.array(ainit) 											#a[i][j] valeur du neurone j de la couche i ,
																		#ce n'est pas une matrice , on initialise aléatoirement
		self.tailles=tailles
		winit=np.empty(len(tailles),dtype=object)	#On initialise le tableau des poids , pour L couches , elle contient L matrices de poids entre la couche L+1 et L
		for l in range(len(tailles)-1):
			#print(tailles[l],tailles[l+1])
			winit[l]=((np.random.rand(tailles[l],tailles[l+1])))*5 		# np.random.rand(i,j) renvoie une matrice de nombres aléatoires de taille i*j
			#print(winit)
		self.w=np.array(winit) #w[L][i,j] est le poids de la connexion entre le neurone i de la couche L et le neurone j de la couche L+1
		
		self.sig=fct_activation		#on appelle sig comme sigma la fonction d'activation du réseau
		self.sigp=drv_activation	#dérivée fournie par l'utilisateur
		
	def propagation(self,entree):
		self.a[0]=copy.deepcopy(entree)
		for l in range(1,len(self.tailles)):
			#print(self.a[l-1],self.w[l])
			self.a[l]=self.sig(np.dot(self.a[l-1],self.w[l-1])) #éventuellement introduire une matrice de biais ici
		return self.a[-1]
		

def sigmoide(x):
	return 1/(1+np.exp(x))
	
def drv_sigmoide(x):
	return 1/(2+np.exp(x)+np.exp(-x))
				
class Vehicules: 
	def __init__(self,position):
		self.position=position
		self.vmax=20
		self.vitesse=0.0
		self.angle=0.0
		self.reseau=Neurones([3,3,4,2],sigmoide,drv_sigmoide)
		self.distance=0.0
		self.vivant=True
		
	def detect_entree(self,dmax):
		gauche,droite,centre=0.0,0.0,0.0
		x,y=self.position
		for k in range(dmax):
			if circuit[x+k][y+k]:
				gauche=k/dmax
		for k in range(dmax):	
			if circuit[x+k][y]:
				centre=k/dmax
		for k in range(dmax):
			if circuit[x+k][y-k]:
				droite=k/dmax
		return np.array([gauche,centre,droite])
	
	def deplacement(self):
		if self.vivant:
			entree=self.detect_entree(10)
			x,y=self.position
			resultat_reseau = self.reseau.propagation(entree)
			#print("res=",resultat_reseau)
			self.vitesse,self.angle= (resultat_reseau[0]+1)*self.vmax/2 , (resultat_reseau[1])*np.pi
			dx,dy=int(self.vitesse*dt*np.cos(self.angle)) , int(self.vitesse*dt*np.sin(self.angle))
			#print("d=",dx,dy)
			self.position= x+dx,y+dy
			
			if circuit[self.position[0]][self.position[1]]:
				self.vivant=False
			else:
				self.distance+=np.sqrt(dx**2+dy**2)
		
		
#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
la_horde=[Vehicules((50,50))for i in range(50)]
for vuatur in la_horde:
	positions=[vuatur.position]
	for k in range(150):
		vuatur.deplacement()
		positions.append(vuatur.position)
		#print(vuatur.distance)
	x_val = [x[0] for x in positions]
	y_val = [x[1] for x in positions]
	plt.plot(x_val,y_val)
plt.show()
