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
#blip normalise, son max vaut 1
def blipn(a,b,x):
    return E(a-x)*E(x-b)*np.exp(1.6/(b-a))    
#version vectorielle
def blipnvect(a,b,x):
    return Evect(a-x)*Evect(x-b)*np.exp(1.6/(b-a))
#derivee
def dblipn(a,b,x):
    return (-dE(a-x)*E(x-b)+E(a-x)*dE(x-b))*np.exp(1.6/(b-a))
#derivee pour la version vectorielle
def dblipnvect(a,b,x):
    return (-dEvect(a-x)*Evect(x-b)+Evect(a-x)*dE(x-b))*np.exp(1.6/(b-a))
#definition du potentiel, deux puits dont le recouvrement est defini par delta
def V(x,y,delta=0.20):
    return -(blipn(0,0.5+delta,x)*blipn(0,0.5+delta,y)+0.5*blipn(0.5-delta,1,x)*blipn(0.5-delta,1,y))

def Vpart(p,delta=0.20):
    return V(p[0],p[1],delta)

def Vvect(x,y,delta=0.20):
    return -(blipnvect(0,0.5+delta,x)*blipnvect(0,0.5+delta,y)+0.5*blipnvect(0.5-delta,1,x)*blipnvect(0.5-delta,1,y))

def gradV(x,y,delta=0.20):
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
            if  (x1-x-i)**2+(y1-y-j)**2<(x1-x)**2+(y1-y)**2:
                x=x+i
                y=y+j
    return x,y

def distance(x1,y1,x2,y2):
    x, y = plusProcheVoisin(x1,y1,x2,y2)
    return math.sqrt((x1-x)**2+(y1-y)**2)

def distanceNormale(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

#potentiel d'interaction radial
def interD(d,delta=0.20):
    return -0.3*blipn(0.25,0.75,d)+0.1*blipn(-0.15,0.15,d)

#norme du gradient radial
def gradInterD(d,delta=0.20):
    return -0.3*dblipn(0.25,0.75,d)+0.1*dblipn(-0.25,0.25,d)

#potentiel a deux particules
def W(x1,y1,x2,y2,delta=0.20):
    d = distance(x1,y1,x2,y2)
    return interD(d, delta)

#allege la notation
def Wpart(p1,p2,delta):
    return W(p1[0],p1[1],p2[0],p2[1])

#force de 2 sur 1
def gradW(x1,y1,x2,y2, delta=0.20):
    x,y = plusProcheVoisin(x1,y1,x2,y2)
    d = distanceNormale(x1,y1,x,y)
    n = math.sqrt((x1-x)**2+(y1-y)**2)
    x = (x1-x)/ n
    y = (y1-y) / n
    #x,y donne le vecteur direction de 2 vers 1
    return gradInterD(d)*x, gradInterD(d)*y

def gradWpart(p1,p2,delta=0.20):
    return gradW(p1[0],p1[1],p2[0],p2[1],delta)


def gen_part(beta,deltat,start,stop, delta=0.15):
	sigma=np.sqrt(2./beta)
	x0=np.asarray([np.random.uniform(),np.random.uniform()])
	path_x =[x0[0]]
	path_y =[x0[1]]
	x_t=x0
	energy = 0
	acc = 0
	for t in np.arange(start,stop,deltat):
		x_temp = x_t  - deltat*gradV(x_t[0],x_t[1], delta) + sigma*np.sqrt(deltat)*np.random.normal(0,1,(2)) 
		x_temp = x_temp % 1 
		energy_temp = V(x_temp[0],x_temp[1],delta)
		ratio = np.exp(-beta*(V(x_temp[0],x_temp[1],delta)-V(x_t[0],x_t[1],delta)))
		ptrans = min(1,ratio)
		temp = np.random.uniform()
		if temp<ptrans:
			x_t=x_temp
			energy +=energy_temp
			acc += 1
		path_x.append(x_t[0])
		path_y.append(x_t[1])
		#print x_t[0],x_t[1]
	energy = energy/acc
	acc = deltat*acc/(stop-start)
	return acc, energy

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
	acc = 0
	energy = 0
	for t in np.arange(start,stop,deltat):
		x1_temp = x1_t - deltat*(gradV(x1_t[0],x1_t[1],delta)+gradw(x1_t, x2_t))\
		 + sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
	 	x2_temp = x2_t - deltat*(gradV(x2_t[0],x1_t[1],delta)+gradw(x2_t,x1_t))\
	 	+ sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
	 	energy_temp = V(x1_temp[0],x1_temp[1],delta)+V(x2_temp[0],x2_temp[1],delta)\
	 	 +w(x1_temp[0],x1_temp[1],x2_temp[0],x2_temp[1])
	 	ratio = np.exp(-beta*(V(x1_temp[0],x1_temp[1],delta)+V(x2_temp[0],x2_temp[1],delta)\
	 	 +w(x1_temp[0],x1_temp[1],x2_temp[0],x2_temp[1])-V(x1_t[0],x1_t[1],delta)-V(x2_t[0],x2_t[1],delta)\
	 	 -w(x1_t[0],x1_t[1],x2_t[0],x2_t[1])))
	 	ptrans = min(1,ratio)
	 	temp = np.random.uniform()
	 	tot+=1
	 	if temp<ptrans:
	 		x1_t = x1_temp
	 		x2_t = x2_temp
	 		acc+=1
	 		energy += energy_temp
 		path_x1.append(x1_t[0])
 		path_y1.append(x1_t[1])
 		path_x2.append(x2_t[0])
 		path_y2.append(x2_t[1])
	energy = energy/acc
	acc = acc*deltat/(stop-start)
	return energy

# ax.plot_surface(x,y,z)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# plt.figure(2)
# plt.contour(x,y,z)
# plt.show()


# delta = 0.20

# betat = np.linspace(0.1, 3, 50)
# tempt = np.linspace(0.1,10,50)
# deltat= 0.001
# start = 0.
# stop = 10.

# energyt = []

# for temp in tempt:

#     acc, energy = gen_part(1/temp,deltat,start,stop,delta)
#     energyt.append(energy)
#     print "Temperature : {}".format(temp)
#     print "Acceptance rate : {}".format(acc)
#     print "Energy : {}".format(energy)

# plt.plot(tempt, energyt)
# plt.show()

# V_path = [V(x,y,delta) for x in path_x for y in path_y ]

# fig = plt.figure(2)
# plt.plot(V_path)

# fig = plt.figure(2)
# plt.hist(path_x)

# fig = plt.figure(3)
# plt.scatter(np.asarray(path_x),np.asarray(path_y))
# plt.axis([0,1,0,1])
# # plt.ion()

# x_plot = np.linspace(0,1,200)
# y_plot = np.linspace(0,1,200)
# x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
# z_plot=Vvect(x_mesh, y_mesh, delta)
# plt.contour(x_plot,y_plot,z_plot)

# fig = plt.figure(4)
# plt.plot(path_x)

# fig = plt.figure(5)
# plt.plot(path_y)

# fig = plt.figure(6)
# ax = fig.add_subplot(111,projection='3d')
# hist, xedges, yedges = np.histogram2d(path_x,path_y, bins=4)

# elements = (len(xedges)-1)*(len(yedges)-1)
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros(elements)
# dx = 0.1 * np.ones_like(zpos)
# dy = dx.copy()
# dz = hist.flatten()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

# plt.show()

# plt.figure(3)
# for i in range(len(path_x)):
# 	if i % 10 == 0:
# 		plt.scatter(path_x[i],path_y[i])
# 		plt.draw()

delta = 0.15
fig = plt.figure(1)
# ax = fig.add_subplot(111, projection = '3d')
x = np.linspace(0,1,200)
y = np.linspace(0,1,200)
x,y=np.meshgrid(x,y)


z = Vvect(x,y,delta)

x_plot = np.linspace(0,1,200)
y_plot = np.linspace(0,1,200)
x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
plt.xlabel("x")
plt.ylabel("y")
z_plot=Vvect(x_mesh, y_mesh, delta)
plt.contour(x_plot,y_plot,z_plot)

plt.show()