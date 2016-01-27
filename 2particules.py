import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math

#convention: 2 particules a 2 dim, matrice
# |x1 x2|
# |y1 y2|


#definition de fonctions dfsdfsdfsdf

def E(x):
	if x < 0 :
		return np.exp(0.4/x)
	else:
		return 0

def Evect(x):
	return np.piecewise(x,[x<0,x>=0], [lambda x: np.exp(0.4/x),0]) 

#derivee de E
def dE(x):
	if x < 0 : 
		return -0.4/x**2*np.exp(0.4/x)
	else:
		return 0

def dEvect(x):
	return np.piecewise(x,[x<0,x>=0],[lambda x: -0.4/x**2*E(x),0])

#la fonction E(a-x)E(x-b) donne un "blip" sur [a;b]
def blip(a,b,x):
	return E(a-x)*E(x-b)

def blipvect(a,b,x):
	return Evect(a-x)*Evect(x-b)

#derivee du blip
def dblip(a,b,x):
	return -dE(a-x)*E(x-b)+E(a-x)*dE(x-b)

def dblipvect(a,b,x):
	return -dEvect(a-x)*Evect(x-b)+Evect(a-x)*dE(x-b)

#blip normalise, son max vaut 1
def blipn(a,b,x):
	return blip(a,b,x)*np.exp(1.6/(b-a))	

def blipnvect(a,b,x):
	return blipvect(a,b,x)*np.exp(1.6/(b-a))

def dblipn(a,b,x):
	return dblip(a,b,x)*np.exp(1.6/(b-a))

def dblipnvect(a,b,x):
	return dblip(a,b,x)*np.exp(1.6/(b-a))
#definition du potentiel, deux puits dont le recouvrement est defini par delta
def V(x,y,delta):
	return -(blipn(0,0.5+delta,x)*blipn(0,0.5+delta,y)+0.5*blipn(0.5-delta,1,x)*blipn(0.5-delta,1,y))

def Vvect(x,y,delta):
	return -(blipnvect(0,0.5+delta,x)*blipnvect(0,0.5+delta,y)+0.5*blipnvect(0.5-delta,1,x)*blipnvect(0.5-delta,1,y))

def gradV(x,y,delta):
	return -np.asarray([dblipn(0,0.5+delta,x)*blipn(0,0.5+delta,y)+0.5*dblipn(0.5-delta,1,x)*blipn(0.5-delta,1,y), \
		dblipn(0,0.5+delta,y)*blipn(0,0.5+delta,x)+0.5*dblipn(0.5-delta,1,y)*blipn(0.5-delta,1,x)])

#definition de la distance entre deux particules, compte tenue de la periodicite

#recherche du "voisin" de la particule 1 le plus proche
def plusProcheVoisin(x1,y1,x2,y2):
	x = x2
	y = y2
	test1 = [-1,0,1]
	test2 = [-1,0,1]
	for i in test1:
		for j in test2:
			if 	(x1-x-i)**2+(y1-y-j)**2<(x1-x)**2+(y1-y)**2:
				x=x+i
				y=y+j
	return x,y

def distance(x1,y1,x2,y2):
	x, y = plusProcheVoisin(x1,y1,x2,y2)
	return math.sqrt((x1-x)**2+(y1-y)**2)

def distanceNormale(x1,y1,x2,y2):
	return math.sqrt((x1-x2)**2+(y1-y2)**2)

def w(x1,y1,x2,y2,k=1,d0=0.10):
	return k/2*(distance(x1,y1,x2,y2)-d0)**2

#force de 2 sur 1
def gradw(part1,part2,k=1,d=0.10):
	x1, y1 = part1[0], part1[1]
	x2, y2 = part2[0], part2[1]
	x, y = plusProcheVoisin(x1,y1,x2,y2)
	#v : vecteur de norme 1 allant de 2 vers 1
	v = [x1-x, x2-x]
	v = v/(math.sqrt(v[0]**2+v[1]**2))
	v = -k(distanceNormale(x1,y1,x,y)-d0)*v
	return v


def gen_part(beta,deltat,start,stop, delta=0.15):
	sigma=np.sqrt(2./beta)
	x0=np.asarray([np.random.uniform(),np.random.uniform()])
	path_x =[x0[0]]
	path_y =[x0[1]]
	x_t=x0
	for t in np.arange(start,stop,deltat):
		x_temp = x_t  - deltat*gradV(x_t[0],x_t[1], delta) + sigma*np.sqrt(deltat)*np.random.normal(0,1,(2)) 
		x_temp = x_temp % 1 
		ratio = np.exp(-beta*(V(x_temp[0],x_temp[1],delta)-V(x_t[0],x_t[1],delta)))
		ptrans = min(1,ratio)
		temp = np.random.uniform()
		if temp<ptrans:
			x_t=x_temp
		path_x.append(x_t[0])
		path_y.append(x_t[1])
		#print x_t[0],x_t[1]
	return path_x,path_y

def gen_2part(beta,deltat,start,stop,delta=0.15):
	sigma=np.sqrt(2./beta)
	x1init=np.asarray([np.random.uniform(),np.random.uniform()])
	x2init=np.asarray([np.random.uniform(),np.random.uniform()])
	path_x1 = [x1init[0]]
	path_y1 = [x1init[1]]
	path_x2 = [x2init[0]]
	path_y2 = [x2init[1]]
	x1_t=x1init
	x2_t=x2init
	for t in np.arange(start,stop,deltat):
		x1_temp = x1_t - deltat*(gradV(x1_t[0],x1_t[1],delta)+gradw(x1_t, x2_t))\
		 + sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
	 	x2_temp = x2_t - deltat*(gradV(x2_t[0],x1_t[1],delta)+gradw(x2_t,x1_t))\
	 	+ sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
	 	ratio = np.exp(-beta*(V(x1_temp[0],x1_temp[1],delta)+V(x2_temp[0],x2_temp[1],delta)\
	 	 +w(x1_temp[0],x1_temp[1],x2_temp[0],x2_temp[1])-V(x1_t[0],x1_t[1],delta)-V(x2_t[0],x2_t[1],delta)\
	 	 -w(x1_t[0],x1_t[1],x2_t[0],x2_t[1])))
	 	ptrans = min(1,ratio)
	 	temp = np.random.uniform()
	 	if temp<ptrans:
	 		x1_t = x1_temp
	 		x2_t = x2_temp
 		path_x1.append(x1_t[0])
 		path_y1.append(x1_t[1])
 		path_x2.append(x2_t[0])
 		path_y2.append(x2_t[1])
	return path_x1, path_y1, path_x2, path_y2



delta = 0.20
beta = 10.
deltat = 0.001
start = 0.
stop = 10.

path_x1, path_y1, path_x2, path_y2 = gen_2part(beta,deltat,start,stop,delta)

fig = plt.figure(1)
plt.axis([0,1,0,1])
plt.ion()

x_plot = np.linspace(0,1,200)
y_plot = np.linspace(0,1,200)
x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
z_plot=Vvect(x_mesh, y_mesh, delta)
plt.contour(x_plot,y_plot,z_plot)

plt.show()

for i in range(len(path_x1)):
	plt.scatter(path_x1[i],path_y1[i], color = 'blue')
	plt.scatter(path_x2[i], path_y2[i], color = 'red')
	plt.draw()
	time.sleep(0.00000000000000001)