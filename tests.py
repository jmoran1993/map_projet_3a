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
def interD(d,delta=0.20):
    return -0.3*blipn(0.25,0.75,d)+0.1*blipn(-0.15,0.15,d)

x=np.arange(0,1,0.01)
b=[for each p in x : interD(p)]
plt.plot(x,b)
plt.xlabel("x")
plt.ylabel("b(x)")
plt.show()
