import math 
import numpy as np 
import random
import matplotlib.pyplot as plt
def potential(x, m=1.0, omega=1.0):
	"""
	Harmonic oscillator potential. 
	Could be modified for other potentials 
	as well
	"""
	return 0.5*m*omega**2*x**2

def grad_potential(x, m=1.0, omega=1.0):
	return x*m*omega**2


beta = 1.0
M = 100
delta_tau = beta/M 
x_max = -4.0
x_min = 4.0

def gen_levy(start, end, worldline):
	"""
	Generate a path between points start and end. 
	This generates a path using the free particle density matrix. 
	Here start and end are indices denoting the positions on the imaginary time slice.
	"""
	x_start = worldline[start]
	levy_path = worldline[:start+1]
	x_end = worldline[end-1]
	N = end-start+1
	for k in range(start+1, end-1):
		delta_tau_prime = (N-k+2)*delta_tau
		x_mean = (delta_tau_prime * levy_path[k - 1] + delta_tau * x_end) / (delta_tau + delta_tau_prime)
		sigma = math.sqrt(1.0 / (1.0 / delta_tau + 1.0 / delta_tau_prime))
		levy_path.append(random.gauss(x_mean, sigma))
	levy_path.append(x_end)
	levy_path[end:] = worldline[end:]
	return levy_path

def create_worldlines(num_particles = 2):
	worldlines = []
	for i in range(num_particles):
		worldline = [(2*np.random.random()-1)*x_max for j in range(M)]
		worldlines.append(worldline)
	return worldlines

def gen_levy_many_particles(start, end, worldlines):
	levy_paths = []
	for i in range(len(worldlines)):
		levy_paths.append(gen_levy(start, end, worldlines[i]))
	return levy_paths

def gen_levy_x(x_start=0.0, x_end=1.0, M=100):
	levy_path = [x_start]
	for k in range(1,M-1):
		delta_tau_prime = (M-k)*delta_tau
		x_mean = (delta_tau_prime * levy_path[k - 1] + delta_tau * x_end) / (delta_tau + delta_tau_prime)
		sigma = math.sqrt(1.0 / (1.0 / delta_tau + 1.0 / delta_tau_prime))
		levy_path.append(random.gauss(x_mean, sigma))
	levy_path.append(x_end)
	return levy_path


