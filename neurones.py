import random
import numpy as np
import copy
import imageio
import matplotlib.pyplot as plt
from multiprocessing import Process #multiprocessing permet de paralléliser les tâches , deja teste avec threading bien moins efficace
#import Array
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
		#print(np.max(self.w[1]))
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
		
		
def mutation(individu,nbmutat):
	for i in range(nbmutat): #on va modifier aléatoirement plusieurs poids du reseau neuronal de l'individu
		couche_random=random.randint(0,len(individu.reseau.w)-2)
		#print('len',len(individu.reseau.w)-1)
		#print('couche',couche_random)
		#print('poids',individu.reseau.w[couche_random])
		#print('type',type(individu.reseau.w[couche_random]))
		i_random=random.randint(0,len(individu.reseau.w[couche_random])-1)
		j_random=random.randint(0,len(individu.reseau.w[couche_random][i_random])-1)
		individu.reseau.w[couche_random][i_random,j_random]=random.random()*2-1

def reproduction(pere,mere):
	enfant=Vehicules((50,50))
	for i in range(len(enfant.reseau.w)): #l'enfant hérite d'une couche sur deux de chaque parent
		if (-1)**i >0 :
			enfant.reseau.w[i]=copy.deepcopy(pere.reseau.w[i])
		else:
			enfant.reseau.w[i]=copy.deepcopy(mere.reseau.w[i])
	return enfant
	
		
def evolution(individus,distances):
	distance_moyenne=sum(distances)/len(distances)
	variance=sum([d**2-distance_moyenne**2 for d in distances])/len(distances)
	print('moy=',distance_moyenne,'\n variance=',variance)
	indices_tries=np.argsort(distances) #argsort renvoie une liste contenant les indices des elements tries par ordre croissant sans modifier la liste
	individus_tries=[individus[k] for k in indices_tries] #les individus sont ici mis dans l'ordre croissant
	taux_mutation=0.1
	taux_meilleurs=0.2 #proportion des meilleurs individus conservés pour la génération suivante
	taux_sauvetage=0.05 #chances qu'un "mauvais" individu soit conservé
	
	#on prend les meilleurs individus
	nombre_meilleurs=int(taux_meilleurs*len(individus_tries))
	parents=individus_tries[nombre_meilleurs:]
	
	#on ajoute quelques individus moins bons au hasard pour la diversité
	for k in range(len(individus_tries[:nombre_meilleurs])):
		if random.random() < taux_sauvetage:
			parents.append(individus_tries[k])
			
	enfants=[]
			
	#on fait muter quelques parents au hasard
	for k in range(len(parents)):
		if random.random() < taux_mutation:
			mutation(parents[k],3)
	
	while len(enfants)<len(individus):
		enfants.append(reproduction(random.choice(parents),random.choice(parents)))
	
	return enfants
	
def sigmoide(x):
	return 1/(1+np.exp(-x))
	
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
		distances=[i*0.0 for i in range (nbangles)] #tableau qui contiendra la distance à un obstacle POUR CHAQUE ANGLE
		angles=[i*np.pi/nbangles for i in range(-int((nbangles)/2),int((nbangles)/2))] #tableau des angles d'observation 
		
		for i in range(len(angles)): #on fixe un angle d'observation , on va regarder dans cette direction
			x,y=self.position		#on copie la position actuelle du véhicule
			for k in range(dmax):	#on itere jusqu'a dmax
				if x>0 and x<xmax and y>0 and y<ymax and not circuit[int(x)][int(y)]: #conditions a laquelle on continue de regarder dans la direction choisie
																					# en gros tant qu'il n'y a pas d'obstacle ou qu'on ne tombe pas sur un mur
					x+=np.cos(self.angle+angles[i])		# x devient x+ projection selon x dans la direction choisie
					y+=np.sin(self.angle+angles[i])		# y devient y+ projection selon y dans la direction choisie
					distances[i]=k/dmax					# la distance pour l'angle d'observation choisi devient k/dmax pour avoir un quotient entre 0 et 1
				
		return np.array(distances) #on convertit en tableau numpy pour que la fonction de propagation puisse calculer rapidement dessus
	
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
		

def generation(la_horde,nb_individus,tracer):		
	#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
	#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
	#la_horde=[Vehicules((50,50))for i in range(nb_individus)]
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
		
		distances_par_vehicule.append(vuatur.distance)
		x_val = [x[0] for x in positions]
		y_val = [x[1] for x in positions]
		if tracer:
			plt.plot(x_val,y_val)
	#(sorted(distances_par_vehicule)[-nb_meilleurs:])
	return la_horde,distances_par_vehicule,[x_val,y_val]
	#plt.show()
	
class Generation_parallele(Process): #necessite de creer une sous classe de Process pour la parallelisation
	def __init__(self,nb_individus,nb_meilleurs):
		Process.__init__(self)
		self.nb_individus = nb_individus 
		self.nb_meilleurs = nb_meilleurs
		self.individus= []
		self.pos_x, self.pos_y=[],[]

	def run(self): #limitations de Process: run() effectue la tache du programme mais ne doit rien retourner
					# c'est pourquoi on utilise une classe dans laquelle on va modifier des propriétés
		nb_individus=self.nb_individus
		nb_meilleurs=self.nb_meilleurs
		
		#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
		#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
		la_horde=[Vehicules((50,50))for i in range(nb_individus)]
		nbvoit_vivantes=len(la_horde)
		
		for k in range(len(la_horde)):
			vuatur=la_horde[k]
			self.individus.append(vuatur)
			positions=[vuatur.position]
			while nbvoit_vivantes>0 and not vuatur.mort():
				if vuatur.mort() :
					nbvoit_vivantes-=1
				else:
					vuatur.deplacement()
					#print(vuatur.reseau.w)
				positions.append(vuatur.position)
				
				#print(vuatur.distance)
			
			self.pos_x.append([x[0] for x in positions])
			self.pos_y.append([x[1] for x in positions])
		#plt.plot(x_val,y_val)
		#print(sorted(distances_par_vehicule)[-nb_meilleurs:])
		#plt.show()

def generation_multithread(nbindividus,nbthreads):
	threads=[Generation_parallele(nbindividus,5) for k in range(nbthreads)]
	for thread in threads:
		thread.start()
	
	for thread in threads:
		print(thread.individus)
		thread.join()
	individus_combines=[]
	for thread in threads:
		print(thread.individus)
		individus_combines+=thread.individus
		
	print(individus_combines)
	"""for thread in threads:
		for k in range(len(thread.pos_x)):
			plt.plot(thread.pos_x[k],thread.pos_y[k])
	plt.show()"""
	
#generation(10,3)
#debut_tps=time.time()
#generation_multithread(40,3)
#print(time.time()-debut_tps)
debut_tps=time.time()
nbind=30
nbgen=10
individus,distances,plot= generation([Vehicules((50,50))for i in range(nbind)],nbind,5)
print(time.time()-debut_tps)
drap=False
for k in range(nbgen):
	if k==nbgen-1:
		drap=True
	debut_tps=time.time()
	new_gen=evolution(individus,distances)
	individus,distances,plot=generation(new_gen,nbind,drap)
	print(time.time()-debut_tps)
#plt.plot(plot[0],plot[1])
plt.show()

#debut_tps=time.time()
#generation(new_gen,40,5)
#print(time.time()-debut_tps)
"""ordres de grandeur de l'interet de la parallelisation:
Processeur: i5-2520m 
120 individus
Programme classique: 9.26s
Programme parallele: 5.05s
"""
