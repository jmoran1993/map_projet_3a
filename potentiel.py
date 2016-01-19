import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

#definition de fonctions

def E(x):
	return np.piecewise(x,[x<0,x>=0], [lambda x: np.exp(0.4/x),0]) 

#la fonction E(a-x)E(x-b) donne un "blip" sur [a;b]
def blip(a,b,x):
	return E(a-x)*E(x-b)

#blip normalise, son max vaut 1
def blipn(a,b,x):
	return blip(a,b,x)*np.exp(4/(b-a))	

#definition du potentiel, deux puits dont le recouvrement est defini par delta
def V(x,y,delta):
	return -(blipn(0,0.5+delta,x)*blipn(0,0.5+delta,y)+0.5*blipn(0.5-delta,1,x)*blipn(0.5-delta,1,y))


delta = 0.15
fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
x = np.linspace(0,1,200)
y = np.linspace(0,1,200)
x,y=np.meshgrid(x,y)


z = V(x,y,delta)

ax.plot_surface(x,y,z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure(2)
plt.contour(x,y,z)
plt.show()