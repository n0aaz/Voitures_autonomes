import random
import numpy as np
import copy
import imageio
import matplotlib.pyplot as plt
#from multiprocessing import Process #multiprocessing permet de paralléliser les tâches , deja teste avec threading bien moins efficace
#import Array
import time
fig1, ax1 = plt.subplots()
fig2, ax2= plt.subplots()

def generer_circuit(image):
	img=imageio.imread(image)
	print(len(img),len(img[0]))
	matrice=np.zeros((len(img),len(img[0])))
	for i in range(len(img)):
		for j in range(len(img[0])):
			r,g,b=tuple(img[i][j][:3])
			if g>100 and b<10 and r<10: #prendre en compte des lignes vertes sur le circuit pour récompenser les bons individus
				matrice[i,j]=2
			elif r+g+b<10:
				matrice[i,j]=1
	print(np.shape(matrice))
	ax1.imshow(np.transpose(matrice))
	return matrice

global no_generation
no_generation =0
normalisation_poids=7
angles_vision=3
distance_vision=25

circuit=generer_circuit('circuit8.png')
#imageio.imwrite('test_circ.png',np.array(circuit))
#print(np.array(circuit[13:85][:100]))
position_initiale=(80,500)
dt=1
xmax,ymax=np.shape(circuit)
#circuit=np.zeros((xmax,ymax))
#circuit[55][50]=1
#for k in range(1000):
#	xx,yy=random.randrange(0,1000),random.randrange(0,1000)
#	circuit[xx][yy]=1
#	plt.plot(xx,yy,marker='o',markersize=4)

class Neurones:
	
	def __init__( self , tailles, fct_activation):
		
		ainit=[[random.random() for k in range(tailles[i])]for i in range(len(tailles))] 
		self.a=np.array(ainit) 											#a[i][j] valeur du neurone j de la couche i ,
																		#ce n'est pas une matrice , on initialise aléatoirement
		self.tailles=tailles
		winit=np.empty(len(tailles),dtype=object)	#On initialise le tableau des poids , pour L couches , elle contient L matrices de poids entre la couche L+1 et L
		for l in range(len(tailles)-1):
			#print(tailles[l],tailles[l+1])
			winit[l]=((np.random.rand(tailles[l],tailles[l+1]))-0.5)*normalisation_poids		# np.random.rand(i,j) renvoie une matrice de nombres aléatoires entre 0 et 1 de taille i*j 
			#print(winit)            #le *2-1 permet de ramener les valeurs entre -1 et 1 
		self.w=np.array(winit) #w[L][i,j] est le poids de la connexion entre le neurone i de la couche L et le neurone j de la couche L+1
		#print(np.max(self.w[1]))
		self.sig=fct_activation		#on appelle sig comme sigma la fonction d'activation du réseau
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
		individu.reseau.w[couche_random][i_random,j_random]+= np.random.normal(scale=normalisation_poids/2)*.999**no_generation #rajout d'un nombre aleatoire decroissant a chaque gen pour affiner
		if individu.reseau.w[couche_random][i_random,j_random]>normalisation_poids:
			individu.reseau.w[couche_random][i_random,j_random]=normalisation_poids
		elif individu.reseau.w[couche_random][i_random,j_random]<-normalisation_poids:
			individu.reseau.w[couche_random][i_random,j_random]=-normalisation_poids
		

def reproduction(pere,mere,nb_modifs):
	enfant=Vehicules(position_initiale)
	choix=random.random()
	#deux opérateurs de reproduction
	#le premier: reproduction barycentrique
	for m in range(nb_modifs):
		
		i=random.choice(range(len(enfant.reseau.w)-1))
		j=random.choice(range(len(enfant.reseau.w[i])))
		k=random.choice(range(len(enfant.reseau.w[i][j])))
		if choix>.5:
			enfant.reseau.w[i][j][k]=(pere.reseau.w[i][j][k]+mere.reseau.w[i][j][k])/2 #reproduction barycentrique
		else:
	#le second: reproduction par copie des poids des parents
			if (-1)**k>0:
				enfant.reseau.w[i][j]=copy.deepcopy(pere.reseau.w[i][j]) #ici l'enfant herite d'un poids sur deux de chaque parent
			else:
				enfant.reseau.w[i][j]=copy.deepcopy(mere.reseau.w[i][j])
		#if (-1)**i >0 :
		#	enfant.reseau.w[i]=copy.deepcopy(pere.reseau.w[i])
		#else:
		#	enfant.reseau.w[i]=copy.deepcopy(mere.reseau.w[i])
	return enfant
	
		
def evolution(individus,distances):
	distance_moyenne=sum(distances)/len(distances)
	variance=sum([d**2-distance_moyenne**2 for d in distances])/len(distances)
	print('moy=',distance_moyenne,'\n variance=',variance)
	indices_tries=np.argsort(distances) #argsort renvoie une liste contenant les indices des elements tries par ordre croissant sans modifier la liste
	individus_tries=[individus[k] for k in indices_tries] #les individus sont ici mis dans l'ordre croissant
	taux_mutation=0.2
	taux_meilleurs=0.6 #proportion des meilleurs individus conservés pour la génération suivante
	taux_sauvetage=0.1 #chances qu'un "mauvais" individu soit conservé
	
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
			mutation(parents[k],2)
	
	while len(enfants)<len(individus):
		enfants.append(reproduction(random.choice(parents),random.choice(parents),5))
	
	return enfants,distance_moyenne,variance
	
def sigmoide(x):
	return 1/(1+np.exp(-x))
	
def tanh(x): #une autre fonction d'activation
	a,b=np.exp(x),np.exp(-x)
	return (a-b)/(a+b)
	
def drv_sigmoide(x):
	return 1/(2+np.exp(x)+np.exp(-x))
				
class Vehicules: 
	def __init__(self,position):
		self.position=position
		self.vmax=15
		self.vitesse=10
		self.angle=0.0#random.random()*2*np.pi
		self.reseau=Neurones([angles_vision,9,1],sigmoide)#modifier le premier coefficient de la liste en raccord avec le nombre de sorties de detect_entree
		self.distance=0.0
		self.vivant=True
		
	def detect_entree(self,dmax,nbangles):
		distances=[i*0.0 for i in range (nbangles)] #tableau qui contiendra la distance à un obstacle POUR CHAQUE ANGLE
		angles=[i*np.pi/nbangles for i in range(-int((nbangles)/2),int((nbangles)/2))] #tableau des angles d'observation 
		
		for i in range(len(angles)): #on fixe un angle d'observation , on va regarder dans cette direction
			x,y=self.position		#on copie la position actuelle du véhicule
			for k in range(dmax):	#on itere jusqu'a dmax
				if x>=0 and x<xmax and y>=0 and y<ymax and not circuit[int(x)][int(y)]==1: #conditions a laquelle on continue de regarder dans la direction choisie
																					# en gros tant qu'il n'y a pas d'obstacle ou qu'on ne tombe pas sur un mur
					x+=np.cos(self.angle+angles[i])		# x devient x+ projection selon x dans la direction choisie
					y+=np.sin(self.angle+angles[i])		# y devient y+ projection selon y dans la direction choisie
					distances[i]=k/dmax					# la distance pour l'angle d'observation choisi devient k/dmax pour avoir un quotient entre 0 et 1
				
		return np.array(distances) #on convertit en tableau numpy pour que la fonction de propagation puisse calculer rapidement dessus
	
	def deplacement(self):
		if self.vivant:
			entree=self.detect_entree(distance_vision,angles_vision)
			#print(entree)
			
			x,y=self.position
			resultat_reseau = self.reseau.propagation(entree)
			#print("res=",resultat_reseau)
			
			#self.vitesse,self.angle= (resultat_reseau[0])*self.vmax , (resultat_reseau[1])*np.pi*2
			self.angle=resultat_reseau*2*np.pi
			dx,dy=int(self.vitesse*dt*np.cos(self.angle)) , int(self.vitesse*dt*np.sin(self.angle))
			#print("d=",dx,dy)
			
			if x<0 or x>=xmax or y<0 or y>=ymax:
				self.distance -=200 #punir les individus qui rentrent dans les murs
			if x<0 or x>=xmax or y<0 or y>=ymax or circuit[x][y] or (dx,dy)==(0,0):#condition pour éviter les blocages, on préferera que les individus soient constamment en mouvement
				if circuit[x][y]==2:
					self.distance+=10000
				self.vivant=False
			else:
				#self.distance+=np.sqrt(dx**2+dy**2)
				self.position= x+dx,y+dy
				xinit,yinit= position_initiale
				self.distance = np.sqrt((xinit-(x+dx))**2+(yinit-(y+dy))**2)
	def mort(self):
		return not self.vivant

def generation(la_horde,nb_individus,tracer):		
	nbvoit_vivantes=len(la_horde)
	distances_par_vehicule=[]
	
	tps=time.time() 
	for k in range(len(la_horde)):
		vuatur=la_horde[k]
		positions=[vuatur.position]
		while nbvoit_vivantes>0 and not vuatur.mort():
			if vuatur.mort() :
				nbvoit_vivantes-=1
			elif (time.time()-tps)>3.0: #introduire une duree de vie limite, certains individus sont sinon capables de ne jamais mourir
				print('je suis mort patron')
				vuatur.vivant=False
			else:
				vuatur.deplacement()
				#print(vuatur.reseau.w)
			positions.append(vuatur.position)
			#print(vuatur.distance)
		
		distances_par_vehicule.append(vuatur.distance)
		x_val = [x[0] for x in positions]
		y_val = [x[1] for x in positions]
		if tracer or (time.time()-tps)>3.0:
			ax1.plot(x_val,y_val,marker='o')
			ax1.set_title("Deplacements")
			ax1.set_xlabel("x")
			ax1.set_ylabel("y")
	#(sorted(distances_par_vehicule)[-nb_meilleurs:])
	return la_horde,distances_par_vehicule,[x_val,y_val]
	#plt.show()
	
#class Generation_parallele(Process): #necessite de creer une sous classe de Process pour la parallelisation
#	def __init__(self,nb_individus,nb_meilleurs):
#		Process.__init__(self)
#		self.nb_individus = nb_individus 
#		self.nb_meilleurs = nb_meilleurs
#		self.individus= []
#		self.pos_x, self.pos_y=[],[]
#
#	def run(self): #limitations de Process: run() effectue la tache du programme mais ne doit rien retourner
#					# c'est pourquoi on utilise une classe dans laquelle on va modifier des propriétés
#		nb_individus=self.nb_individus
#		nb_meilleurs=self.nb_meilleurs
#		
#		#N=Neurones([2,3,4,2],sigmoide,drv_sigmoide)
#		#print("resultat de la propagation",N.propagation(np.array([18,2])),"activations=",N.a)
#		la_horde=[Vehicules(position_initiale)for i in range(nb_individus)]
#		nbvoit_vivantes=len(la_horde)
#		
#		for k in range(len(la_horde)):
#			vuatur=la_horde[k]
#			self.individus.append(vuatur)
#			positions=[vuatur.position]
#			while nbvoit_vivantes>0 and not vuatur.mort():
#				if vuatur.mort() :
#					nbvoit_vivantes-=1
#				else:
#					vuatur.deplacement()
#					#print(vuatur.reseau.w)
#				positions.append(vuatur.position)
#				
#				#print(vuatur.distance)
#			
#			self.pos_x.append([x[0] for x in positions])
#			self.pos_y.append([x[1] for x in positions])
#		#plt.plot(x_val,y_val)
#		#print(sorted(distances_par_vehicule)[-nb_meilleurs:])
#		#plt.show()
#
#def generation_multithread(nbindividus,nbthreads):
#	threads=[Generation_parallele(nbindividus,5) for k in range(nbthreads)]
#	for thread in threads:
#		thread.start()
#	
#	for thread in threads:
#		print(thread.individus)
#		thread.join()
#	individus_combines=[]
#	for thread in threads:
#		print(thread.individus)
#		individus_combines+=thread.individus
#		
#	print(individus_combines)
#	"""for thread in threads:
#		for k in range(len(thread.pos_x)):
#			plt.plot(thread.pos_x[k],thread.pos_y[k])
#	plt.show()"""
#	
#generation(10,3)
#debut_tps=time.time()
#generation_multithread(40,3)
#print(time.time()-debut_tps)
debut_tps=time.time()
nbind=50
nbgen=20
individus,distances,plot= generation([Vehicules(position_initiale)for i in range(nbind)],nbind,5)
print(time.time()-debut_tps)
drap=False
liste_dist=[]
liste_var=[]
for k in range(nbgen):
	no_generation=k
	if k==nbgen-1:
		drap=True
	debut_tps=time.time()
	new_gen,dmoy,variance=evolution(individus,distances)
	individus,distances,plot=generation(new_gen,nbind,drap)
	liste_dist.append(dmoy)
	liste_var.append(variance)
	#liste_temps.append(time.time()-debut_tps)
	#if time.time()-debut_tps>3:
	#	break
	print('generation_numero:',no_generation,'temps=',time.time()-debut_tps)
#plt.plot(plot[0],plot[1])
ax1.grid(True)
#fig2.subplot(211)
ax2.plot(range(no_generation+1),liste_dist)
#fig2.subplot(212)
#ax2.plot(range(no_generation+1),liste_var)
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
