import random
import numpy as np
import copy
import imageio
import matplotlib.pyplot as plt
from multiprocessing import Process #multiprocessing permet de paralléliser les tâches , deja teste avec threading bien moins efficace
import time


dt=1/5
xmax,ymax=1000,1000
circuit=np.zeros((xmax,ymax))
circuit[55][50]=1
for k in range(1000):
	xx,yy=random.randrange(0,1000),random.randrange(0,1000)
	circuit[xx][yy]=1
	plt.plot(xx,yy,marker='o',markersize=4)

class Neurones:
	
	def __init__( self , tailles, fct_activation, drv_activation):
		
		ainit=[[random.random() for k in range(tailles[i])]for i in range(len(tailles))] # *2-1 pour ramener sur [-1;1]
		self.a=np.array(ainit) 											#a[i][j] valeur du neurone j de la couche i ,
																		#ce n'est pas une matrice , on initialise aléatoirement
		self.tailles=tailles
		winit=np.empty(len(tailles),dtype=object)	#On initialise le tableau des poids , pour L couches , elle contient L matrices de poids entre la couche L+1 et L
		for l in range(len(tailles)-1):
			#print(tailles[l],tailles[l+1])
			winit[l]=((np.random.rand(tailles[l],tailles[l+1])))*2-1 		# np.random.rand(i,j) renvoie une matrice de nombres aléatoires entre 0 et 1 de taille i*j 
			#print(winit)            le *2-1 permet de ramener les valeurs entre -1 et 1 
		self.w=np.array(winit) #w[L][i,j] est le poids de la connexion entre le neurone i de la couche L et le neurone j de la couche L+1
		
		self.sig=fct_activation		#on appelle sig comme sigma la fonction d'activation du réseau
		self.sigp=drv_activation	#dérivée fournie par l'utilisateur
		self.memo={}	#memoisation pour ameliorer les performances
		
	def propagation(self,entree):
		if tuple(entree) not in self.memo: #si la valeur du calcul n'est pas connue on la calcule
			"""Attention: l'entree sera une liste/un tableau ne pouvant pas etre utilise comme cle de dictionnaire
			c'est pourquoi il faut le transformer d'abord en tuple"""
			self.a[0]=copy.deepcopy(entree)
			for l in range(1,len(self.tailles)):
				#print(self.a[l-1],self.w[l])
				self.a[l]=self.sig(np.dot(self.a[l-1],self.w[l-1])) #éventuellement introduire une matrice de biais ici
			self.memo[tuple(entree)]=self.a[-1] #on ajoute la valeur calculee au dictionnaire pour une eventuelle utilisation future 
			return self.a[-1]
		else:
			return self.memo[tuple(entree)]
		

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
		self.reseau=Neurones([5,4,3,2],sigmoide,drv_sigmoide)#modifier le premier coefficient de la liste en raccord avec le nombre de sorties de detect_entree
		self.distance=0.0
		self.vivant=True
		
	def detect_entree(self,dmax,nbangles):
		distances=[i*0.0 for i in range (nbangles)]
		angles=[i*np.pi/nbangles for i in range(-int((nbangles)/2),int((nbangles)/2))] #modifier ici les angles
		
		for i in range(len(angles)):
			x,y=self.position
			d=0 
			for k in range(dmax):
				if x>0 and x<xmax and y>0 and y<ymax and not circuit[int(x)][int(y)]:
					x+=np.cos(self.angle+angles[i])
					y+=np.sin(self.angle+angles[i])
					distances[i]=k/dmax
				
		return np.array(distances)
	
	def deplacement(self):
		if self.vivant:
			entree=self.detect_entree(10,5)
			x,y=self.position
			resultat_reseau = self.reseau.propagation(entree)
			#print("res=",resultat_reseau)
			self.vitesse,self.angle= (resultat_reseau[0]+1)*self.vmax/2 , (resultat_reseau[1])*np.pi
			dx,dy=int(self.vitesse*dt*np.cos(self.angle)) , int(self.vitesse*dt*np.sin(self.angle))
			#print("d=",dx,dy)
			
			if x<0 or x>=xmax or y<0 or y>=ymax or circuit[x][y]:
				self.vivant=False
			else:
				self.distance+=np.sqrt(dx**2+dy**2)
				self.position= x+dx,y+dy
				
	def mort(self):
		return not self.vivant
		

def generation(nb_individus,nb_meilleurs):		
	#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
	#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
	la_horde=[Vehicules((50,50))for i in range(nb_individus)]
	nbvoit_vivantes=len(la_horde)
	distances_par_vehicule=[]
	
	for k in range(len(la_horde)):
		vuatur=la_horde[k]
		positions=[vuatur.position]
		while nbvoit_vivantes>0 and not vuatur.mort():
			if vuatur.mort() :
				nbvoit_vivantes-=1
			else:
				vuatur.deplacement()
				#print(vuatur.reseau.w)
			positions.append(vuatur.position)
			#print(vuatur.distance)
		
		distances_par_vehicule.append((vuatur.distance,k))
		x_val = [x[0] for x in positions]
		y_val = [x[1] for x in positions]
		#plt.plot(x_val,y_val)
	print(sorted(distances_par_vehicule)[-nb_meilleurs:])
	#plt.show()
	
class Generation_parallele(Process): #necessite de creer une sous classe de Process pour la parallelisation
	def __init__(self,nb_individus,nb_meilleurs):
		Process.__init__(self)
		self.nb_individus = nb_individus 
		self.nb_meilleurs = nb_meilleurs
		self.individus= []
		self.distances=[]
		self.pos_x, self.pos_y=[],[]

	def run(self): #limitations de Process: run() effectue la tache du programme mais ne doit rien retourner
					# c'est pourquoi on utilise une classe dans laquelle on va modifier des propriétés
		nb_individus=self.nb_individus
		nb_meilleurs=self.nb_meilleurs
		
		#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
		#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
		la_horde=[Vehicules((50,50))for i in range(nb_individus)]
		nbvoit_vivantes=len(la_horde)
		distances_par_vehicule=[]
		
		for k in range(len(la_horde)):
			vuatur=la_horde[k]
			positions=[vuatur.position]
			while nbvoit_vivantes>0 and not vuatur.mort():
				if vuatur.mort() :
					nbvoit_vivantes-=1
				else:
					vuatur.deplacement()
					#print(vuatur.reseau.w)
				positions.append(vuatur.position)
				#print(vuatur.distance)
			
			distances_par_vehicule.append((vuatur.distance,k))
			self.pos_x.append([x[0] for x in positions])
			self.pos_y.append([x[1] for x in positions])
		self.distances=distances_par_vehicule
		#plt.plot(x_val,y_val)
		#print(sorted(distances_par_vehicule)[-nb_meilleurs:])
		#plt.show()

def generation_multithread(nbindividus,nbthreads):
	threads=[Generation_parallele(nbindividus,5) for k in range(nbthreads)]
	for thread in threads:
		thread.start()
	
	for thread in threads:
		thread.join()
	"""for thread in threads:
		for k in range(len(thread.pos_x)):
			plt.plot(thread.pos_x[k],thread.pos_y[k])
	plt.show()"""
	
#generation(10,3)
debut_tps=time.time()
generation_multithread(40,3)
print(time.time()-debut_tps)
debut_tps=time.time()
generation(120,5)
print(time.time()-debut_tps)
"""ordres de grandeur de l'interet de la parallelisation:
Processeur: i5-2520m 
120 individus
Programme classique: 9.26s
Programme parallele: 5.05s
"""
