#!/usr/bin/env python
# coding: utf-8


import os
import sys
# Add parent directory in order to load in GA.py and QN.py
sys.path.append(os.pardir)
from GeneticAlgorithms import PWCGeneticAlgorithm
from QuantumNetworkEnvsClean import QuantumNetworkPWC
from joblib import Parallel, delayed
import numpy as np
import qutip as qt
from qutip import tensor, ket, basis
import pickle
import matplotlib.pyplot as plt
import time


# theta1 = np.pi/2
# theta2 = np.pi/2
# theta3 = np.pi/2
# theta4 = np.pi/2
# init_state = qt.ket('0000')
# plusX4 = qt.hadamard_transform(N=4)*init_state 
# CP1 = qt.cphase(theta1, N=4, control=0, target=1)
# CP2 = qt.cphase(theta2, N=4, control=1, target=2)
# CP3 = qt.cphase(theta3, N=4, control=2, target=3)
# CP4 = qt.cphase(theta4, N=4, control=3, target=0)
# cluster = CP1 * CP2 * CP3 * CP4 * plusX4
GHZ3 = (qt.ket("000")+qt.ket("111"))/np.sqrt(2)
plus = (qt.ket('0')+qt.ket('1'))/np.sqrt(2)

# define the path
path = os.getcwd()+'/Optimal_controls/GHZ3/Trial2/'
# define target state of the qubit network, number of qubits, cav dim via k+1, num pwc steps
# population size of CGA algo and init_state str

N=3
cav_dim=5
init_state = ket('100')
steps=10
population_size=12


env = QuantumNetworkPWC(N_qubits=N, N_steps=steps,
                         max_time=10, cav_dim=cav_dim,
                         targ_state=GHZ3, max_driving=1, max_coupling=1,
                         init_state=init_state,
                        resolution=6, save_path=path, sharpness=2.2)
GA = PWCGeneticAlgorithm(N_steps=steps+1, num_distinct_controls=(N+1),
                        population_size=population_size)
population = GA.initial_population()


max_count=30000
bench=0
done = False
count=0
base_rate=0.25
checkpoint_count=0
time_a = time.time()
while not done:
    fitness_ls = Parallel(n_jobs=6)(delayed(env.step)(pop, his=True, Noise=False) for pop in population)
    fitness = np.array(fitness_ls)
#     fitness_ls=[]
#     for i in range(population_size):
#         fit = env.step(population[i], his=True, Noise=True)
#         fitness_ls.append(fit)
#     fitness=np.array(fitness_ls)
    max_fit = np.max(fitness)
    if max_fit>bench:
        print(" EPISODE: {},check_point: {}, average cost: {}, Best Cost: {}".format(count,checkpoint_count,
                                                                                     np.round(np.average(fitness), decimals=4), 
                                                                                              np.max(fitness)))
        if os.path.isdir(path+'Checkpoint_{}'.format(checkpoint_count))==False:
            os.makedirs(path+'Checkpoint_{}'.format(checkpoint_count))
        bench=max_fit
        best_index = np.argmax(fitness)
        best_chromosome = population[best_index,:]
        env.plot_initializer(chromosome=best_chromosome, Noise=False, alt_path_ext='Checkpoint_{}'.format(checkpoint_count))
        env.save(chromosome=best_chromosome)
        checkpoint_count+=1
    if max_fit>1.95 or count==max_count:
        done=True

        break
    survivors, survivor_fitness = GA.darwin(population,fitness)
    mothers, fathers = GA.pairing(survivors, survivor_fitness)
    offspring = GA.mating_procedure(ma=mothers, da=fathers)
    unmutated_next_gen = GA.build_next_gen(survivors,offspring)
    d_next_gen = GA.build_next_gen(survivors,offspring)
    rate = max(base_rate*np.exp(-0.0005*count),0.1)
    mutated_next_gen = GA.mutate(unmutated_next_gen, rate=rate)
    population = mutated_next_gen
    count+=1
time_b = time.time()
print("Time: ", (time_b-time_a)/max_count)



