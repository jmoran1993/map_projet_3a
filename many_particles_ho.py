import math 
import numpy as np 

def potential(x, m=1.0, omega=1.0):
	"""
	Harmonic oscillator potential. 
	Could be modified for other potentials 
	as well
	"""
	return 0.5*m*omega**2*x**2

def grad_potential(x, m=1.0, omega=1.0):
	return x*m*omega**2

## global parameters

tau = 10.0 ## imaginary time period
M = 100 ## Number of time slices 
delta_tau = tau/M ## imaginary time step

n_bins = 100 ## for histogram 
delta = 1.0 ## metropolis step size

mc_steps = 100000 ## Number of monte-carlo steps 
x_max = 4.0 ## We restrict the positions between +4 and -4
x_min = -4.0

##Random Initialization of the initial positions of the paths

pos_x = [(2*np.random.random()-1)*x_max for j in range(M)]

def metropolis_step(pos_x_new, pos_x):
	j = int(np.random.random()*M)
	j_minus = j-1
	j_plus = j+1
	if j_minus < 0:
		j_minus = M-1
	if j_plus > M-1:
		j_plus = 0
	pos_x_temp = pos_x[j] + (2*np.random.random()-1)*delta
	delta_E = potential(pos_x_temp) - potential(pos_x[j]) + \
				0.5*((pos_x[j_plus]-pos_x_temp)/delta_tau)**2 + \
				0.5*((pos_x_temp-pos_x[j_minus])/delta_tau)**2 - \
				0.5*((pos_x[j_plus] - pos_x[j])/delta_tau)**2 - \
				0.5*((pos_x[j]-pos_x[j_minus])/delta_tau)**2
	if (delta_E < 0.0) | (np.exp(-delta_tau*delta_E) > np.random.random()):		
		pos_x[j] = pos_x_temp
		pos_x_new[0] = pos_x_temp
		return True
	else:
		pos_x_new[0] = pos_x[j]
		return False

## Thermalization step 

thermal_steps = 5
accepted_steps = 0

##I have to pass a mutable variable into the function metropolisy step 
## Thus I take a one-element list to construct the new position 
## Thermalization is performed to start with some random distribution of 
## positions, not with positions that are uniformly distributed
pos_x_new = [0.0]

print "Performing Thermalization"
for step in range(thermal_steps):
	for j in range(M):
		if metropolis_step(pos_x_new, pos_x):
			accepted_steps +=1

##Acceptance rate 

print "Acceptance Rate {}".format(accepted_steps*1.0/(M*thermal_steps)*100)

psi_2 = np.zeros((M)) ##Probability distribution of the particle 

print "Performing monte-carlo steps"
energy_sum = 0.0
energy_squared_sum = 0.0 
accepted_steps = 0
for step in range(mc_steps):
	for j in range(M):
		if metropolis_step(pos_x_new, pos_x):
			accepted_steps +=1
		bin_pos = (pos_x_new[0]-x_min)/(x_max-x_min) * n_bins
		if (bin_pos >=0) & (bin_pos < M):
			psi_2[bin_pos]  +=1
		energy = potential(pos_x_new[0]) + 0.5*pos_x_new[0]*grad_potential(pos_x_new[0])
		energy_sum += energy 
		energy_squared_sum += energy*energy

print "Acceptance Rate {}".format(accepted_steps*1.0/(M*thermal_steps)*100)

steps = mc_steps*M 
energy_average = energy_sum/steps 

print "Average energy calculated {}".format(energy_average)

energy_variance = energy_squared_sum/steps - energy_average*energy_average


print "Standard Deviation in Energy {}".format(np.sqrt(energy_variance/steps))

## Plot for wavefunction
energy_average_psi = 0.0
x_list = []
psi_list = []
dx = (x_max-x_min)/n_bins
for i in range(n_bins):
    x = x_min + dx*(i+0.5)
    x_list.append(x)
    psi_list.append(psi_2[i]/steps)
    energy_average_psi += psi_2[i]/steps * (0.5*x*grad_potential(x) + potential(x))

print "Average energy from psi^2 {}".format(energy_average_psi)


energy_average_theory = 0.5+ 1/(np.exp(tau)-1)
print "Average energy theoretical {}".format(energy_average_theory)
import matplotlib.pyplot as plt 
plt.scatter(np.asarray(x_list), np.asarray(psi_list))



		  
