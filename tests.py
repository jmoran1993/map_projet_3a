import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math


def dminsq(x1,x2):
	return min((x1-x2)**2,(x1-x2-1)**2,(x1-x2+1)**2)

def dist(x1,y1,x2,y2):
	return math.sqrt(dminsq(x1,x2)+dminsq(y1,y2))

tests = [[0,0,0,0],[0,0,1,1],[0.25,0.85,0.5,0.1]]

for x in tests:
	print dist(x[0],x[1],x[2],x[3])

print 1/math.sqrt(8)