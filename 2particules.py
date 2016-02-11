import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math



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

def gen2part(beta,deltat,start,stop,delta=0.15):
    sigma = np.sqrt(2./beta)
    p1_0=np.asarray([np.random.uniform(),np.random.uniform()])
    p2_0=np.asarray([np.random.uniform(),np.random.uniform()])
    path_x1 = [p1_0[0]]
    path_y1 = [p1_0[1]]
    path_x2 = [p2_0[0]]
    path_y2 = [p2_0[1]]
    p1_t = p1_0
    p2_t = p2_0
    acc = 0
    energy = 0
    for t in np.arange(start,stop,deltat):
        p1_temp = p1_t-deltat*(gradV(p1_t[0],p1_t[1],delta)+gradWpart(p1_t,p2_t,delta))\
            +sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
        p1_temp[0], p1_temp[1] = p1_temp[0] % 1, p1_temp[1] % 1
        p2_temp = p2_t-deltat*(gradV(p2_t[0],p2_t[1],delta)+gradWpart(p2_t,p1_t,delta))\
            +sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))
        p2_temp[0], p2_temp[1] = p2_temp[0] % 1, p2_temp[1] % 1    
        energy_temp = Vpart(p1_temp,delta)+Vpart(p2_temp,delta)+Wpart(p1_temp,p2_temp,delta)
        energy_t = Vpart(p1_t,delta)+Vpart(p2_t,delta)+Wpart(p1_t,p2_t,delta)
        ratio = np.exp(-beta*(Vpart(p1_temp,delta)+Vpart(p2_temp,delta)+Wpart(p1_temp,p2_temp,delta)\
            -(Vpart(p1_t,delta)+Vpart(p2_t,delta)+Wpart(p1_t,p2_t,delta) ) ) )        
        ptrans = min(1,ratio)
        temp=np.random.uniform()
        if temp<ptrans:
            p1_t = p1_temp
            p2_t = p2_temp
            acc += 1
            energy += energy_temp
        path_x1.append(p1_t[0])
        path_y1.append(p1_t[1])
        path_x2.append(p2_t[0])
        path_y2.append(p2_t[1])
    energy = energy/acc
    acc = acc*deltat / (stop-start)
    return path_x1, path_y1, path_x2, path_y2, acc, energy


delta = 0.20

beta = 15
deltat= 0.001
start = 0.
stop = 10.

path_x1, path_y1, path_x2, path_y2, acc, energy = gen2part(beta,deltat,start,stop,delta)

print "Acceptance rate : {}".format(acc)
print "Energy : {}".format(energy)
fig = plt.figure(1)
#plt.scatter(np.asarray(path_x1),np.asarray(path_y1), color='blue')
#plt.scatter(np.asarray(path_x2),np.asarray(path_y2), color='red')
plt.axis([0,1,0,1])
x_plot = np.linspace(0,1,200)
y_plot = np.linspace(0,1,200)
x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
z_plot=Vvect(x_mesh, y_mesh, delta)
plt.contour(x_plot,y_plot,z_plot)
plt.ion()
text = plt.text(0.1,0.9, "Iteration : ")

plt.show()

plt.figure(1)
for i in range(len(path_x1)):
  if i % 50 == 0:
    plt.scatter(path_x1[i],path_y1[i], color='green')
    plt.scatter(path_x2[i],path_y2[i], color='red')
    text.set_text("Iteration : "+str(i))
    plt.draw()