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

def quantInter(p1, p2, beta, sigma):
    d = distance(p1[0],p1[1],p2[0],p2[1])
    return 1/(2*beta*sigma**2)*d**2

def quantInterGrad(p1,p2,beta,sigma):
    x,y = plusProcheVoisin(p1[0],p1[1],p2[0],p2[1])
    f1 =  -1/(4*beta*sigma**2)*(p1[0]-x)
    f2 = -1/(4*beta*sigma**2)*(p1[1]-y)
    return f1,f2

def gen2partQ(beta,deltat,start,stop,delta=0.15, M=10):
    sigma = np.sqrt(M*2./beta)
    #Definition des deux particules 
    p1_0 = []
    p2_0 = []
    p1 = [np.random.uniform(), np.random.uniform()]
    p2 = [np.random.uniform(), np.random.uniform()]
    for i in range(M):
        # p1_0.append([np.random.uniform(),np.random.uniform()])
        # p2_0.append([np.random.uniform(),np.random.uniform()])
        p1_0.append(p1)
        p2_0.append(p2)
    # p1[M-1]=p1[0]
    # p2[M-1]=p2[0]
    path_x1 = []
    path_y1 = []
    path_x2 = []
    path_y2 = []
    for path in (path_x1, path_y1, path_x2, path_y2):
        for i in range(M):
            path.append([])
    print len(path_x1)
    p1_t=p1_0
    p2_t=p2_0
    acc = 0 
    energy = 0
    #p1_t[i] = coordonnees de la ieme tranche de la 1ere particule au temps t
    for t in np.arange(start,stop,deltat):
        p1_temp = np.zeros((M,2))
        p2_temp = np.zeros((M,2))
        for i in range(M):
            p1_temp[i]=(p1_t[i]-deltat*(gradV(p1_t[i][0],p1_t[i][1],delta)+gradWpart(p1_t[i],p2_t[i],delta))\
            +sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))\
            -deltat*np.asarray(quantInterGrad(p1_t[i],p1_t[(i+1)%M],beta,sigma)))%1

            p2_temp[i]=(p2_t[i]-deltat*(gradV(p1_t[i][0],p1_t[i][1],delta)+gradWpart(p2_t[i],p1_t[i],delta))\
            +sigma*np.sqrt(deltat)*np.random.normal(0,1,(2))\
            -deltat*np.asarray(quantInterGrad(p2_t[i],p2_t[(i+1)%M],beta,sigma)))%1
        energy_t = 0
        energy_temp = 0
        energie = 0 
        for i in range(M):
            energy_t = energy_t + Vpart(p1_t[i],delta)+Vpart(p2_t[i],delta)+Wpart(p1_t[i],p2_t[i],delta)\
            +quantInter(p1_t[i],p1_t[(i+1)%M],beta,sigma)+quantInter(p2_t[i],p2_t[(i+1)%M],beta,sigma)

            energy_temp = energy_temp + Vpart(p1_temp[i],delta)+Vpart(p2_temp[i],delta)+Wpart(p1_temp[i],p2_temp[i],delta)\
            +quantInter(p1_temp[i],p1_temp[(i+1)%M],beta,sigma)+quantInter(p2_temp[i],p2_temp[(i+1)%M],beta,sigma)

            energie = energie+ Vpart(p1_temp[i],delta)+Vpart(p2_temp[i],delta)+Wpart(p1_temp[i],p2_temp[i],delta)

        ratio = np.exp(-beta*(energy_temp-energy_t))
        ptrans = min(1,ratio)
        temp = np.random.uniform()
        if temp < ptrans:
            p1_t = p1_temp
            p2_t = p2_temp
            acc += 1
            energy += energie
        for i in range(M):
            path_x1[i].append(p1_t[i][0])
            path_y1[i].append(p1_t[i][1])
            path_x2[i].append(p2_t[i][0])
            path_y2[i].append(p2_t[i][1])
    energy = energy/(acc*M)
    acc = acc*deltat / (stop-start)

    return path_x1, path_y1, path_x2, path_y2, acc,energy


delta = 0.20

beta = 10
deltat= 0.001
start = 0.
stop = 10.

path_x1, path_y1, path_x2, path_y2,acc,energy = gen2partQ(beta,deltat,start,stop,delta)

print("Acceptance rate : {}").format(acc)
print("Average Energy : {}").format(energy)

fig = plt.figure(1)
plt.axis([0,1,0,1])
x_plot = np.linspace(0,1,200)
y_plot = np.linspace(0,1,200)
plt.xlabel("x")
plt.ylabel("y")
x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
z_plot=Vvect(x_mesh, y_mesh, delta)
plt.contour(x_plot,y_plot,z_plot)
plt.scatter(path_x1[1],path_y1[1], color='blue')
plt.scatter(path_x2[1],path_y2[1], color='red')
plt.show()