#!/usr/bin/env python
# coding: utf-8

# In[261]:


import numpy as np


class QCGeneticAlgorithm:

    def __init__(self, basis_dim, num_distinct_controls, num_basis_params,  population_size):


        "define parameters"
        self.N_pop = population_size
        self.N_controls = num_distinct_controls
        self.N_var = basis_dim * num_basis_params * num_distinct_controls


    "Initialize a first population"
    def initial_population(self):
        return np.random.uniform(-1,1, (self.N_pop, self.N_var))


    "cost will be problem specific"

    "Select best N/2 to survive!"
    def darwin(self, population, fitness):
        #takes the current generation of solutions (population) and kills the weakest half
        # Returns the ordered survivors (with the elite survivor at index 0) and their costs
        # np.argsort sorts the indices from smallest as zeroth element to largest thus to get the largest elem at 0
        # we pass -1 X our array to be sorted, then take the first
        indices = np.argsort(-fitness)[:int(self.N_pop/2)]
        survivors = population[indices]
        survivor_fitness = fitness[indices]
        return survivors, survivor_fitness

    def pairing(self, survivors, survivor_fitness):
        # Select pair of chromosomes to mate from the survivors
        prob_dist = survivor_fitness/np.sum(survivor_fitness)
        ma_indices = np.random.choice(np.arange(survivors.shape[0]),
                                   size=int(self.N_pop/4), p=prob_dist, replace=True)
        da_indices = np.zeros_like(ma_indices)
        index_ls = np.arange(survivors.shape[0])
        for i, ma in enumerate(ma_indices):
            inter_prob_dist = prob_dist.copy()
            inter_prob_dist[ma] = 0.
            da_indices[i] = np.random.choice(index_ls, size=1, p=inter_prob_dist)
        # da_indices = np.random.choice(np.arange(survivors.shape[0]),
#                                    size=int(self.N_pop/4), p=prob_dist, replace=False)
        father_chromes = survivors[da_indices]
        mother_chromes = survivors[ma_indices]
        return mother_chromes, father_chromes

    def mating_procedure(self, ma, da):
        # separate the ma and da chromosomes into respective control pulses
        ma_controls = np.hsplit(ma, self.N_controls)
        da_controls = np.hsplit(da, self.N_controls)
        num_parents = ma.shape[0]
        # Generate random numbers to probabilistically swap entire controls between
        # parent offspring
        swap_prob = np.random.normal(loc=0.5,scale=0.2,	size = self.N_controls)
        random_number = np.random.uniform(0,1,size = self.N_controls)
        # Loop over all possible control as probabilistically implement a swap
        off_1_ls = []
        off_2_ls = []
        off_1_ls.append(ma_controls[0])
        off_2_ls.append(da_controls[0])
        for idx in range(1,self.N_controls):
            if random_number[idx]>swap_prob[idx]:
                off_1_ls.append(da_controls[idx])
                off_2_ls.append(ma_controls[idx])
            else:
                off_1_ls.append(ma_controls[idx])
                off_2_ls.append(da_controls[idx])
        # Turn lists of jumbled controls into full arrays
        off_1 = np.concatenate(off_1_ls, axis=1)
        off_2 = np.concatenate(off_2_ls, axis=1)
        # define an array of combination parameters beta to use to combine the parent chromosomes
        # in a continuous way
        beta = np.random.uniform(0,1, size=(num_parents, self.N_var))
        beta_inverse = 1 - beta
        # Randomly select an array of ones to mask the beta array (this also randomly selects)
        # the indices to be swapped
        swap_array = np.random.randint(low=0,high=2,size=(num_parents, self.N_var))
        masked_beta = swap_array * beta
        masked_inverse_beta = swap_array * beta_inverse
        not_swap_array = np.mod((swap_array + 1),2)
        # Implement the b*O1 + (1-b)*02 on each chosen element
        new_off_1 = masked_beta * off_1 + masked_inverse_beta * off_2 + not_swap_array * off_1
        new_off_2 = masked_beta * off_2 + masked_inverse_beta * off_1 + not_swap_array * off_2
        offspring_array = np.concatenate((new_off_1,new_off_2),axis=0)
        return offspring_array


    def build_next_gen(self, survivors, offspring):
        # build next generation
        return np.concatenate((survivors, offspring), axis=0)


    def mutate(self, population, rate=0.3):
        # Mutate the new generation
        number_of_mutations = int((population.shape[0] - 1) * population.shape[-1] * rate)
        row_indices = np.random.choice(np.arange(1,int((population.shape[0]))), size=number_of_mutations)
        col_indices = np.random.choice(np.arange(0,int((population.shape[-1]))), size=number_of_mutations)
        mutated_population = np.copy(population)
        #mutated_population[row_indices, col_indices] = np.random.uniform(-1,1,size=number_of_mutations)
        mutated_population[row_indices, col_indices] = np.clip(population[row_indices, col_indices] + np.random.normal(loc=0, scale=0.2, size=number_of_mutations), a_min=-1, a_max=1)
        return mutated_population


class PWCGeneticAlgorithm:

    def __init__(self,N_steps, num_distinct_controls,  population_size):


        "define parameters"
        self.N_pop = population_size
        self.N_controls = num_distinct_controls
        self.N_var = N_steps * num_distinct_controls


    "Initialize a first population"
    def initial_population(self):
        return np.random.uniform(-1,1, (self.N_pop, self.N_var))


    "cost will be problem specific"

    "Select best N/2 to survive!"
    def darwin(self, population, fitness):
        #takes the current generation of solutions (population) and kills the weakest half
        # Returns the ordered survivors (with the elite survivor at index 0) and their costs
        # np.argsort sorts the indices from smallest as zeroth element to largest thus to get the largest elem at 0
        # we pass -1 X our array to be sorted, then take the first
        indices = np.argsort(-fitness)[:int(self.N_var/2 - 1)]
        survivors = population[indices]
        survivor_fitness = fitness[indices]
        return survivors, survivor_fitness

    def pairing(self, survivors, survivor_fitness):
        # Select pair of chromosomes to mate from the survivors
        positive_sf = survivor_fitness + np.abs(np.min(survivor_fitness))
        prob_dist = (positive_sf)/np.sum(positive_sf)
        ma_indices = np.random.choice(np.arange(survivors.shape[0]),
                                   size=int(self.N_pop/4), p=prob_dist, replace=True)
        da_indices = np.zeros_like(ma_indices)
        index_ls = np.arange(survivors.shape[0])
        for i, ma in enumerate(ma_indices):
            inter_prob_dist = prob_dist.copy()
            inter_prob_dist[ma] = 0.
            inter_prob_dist_normd = inter_prob_dist/(np.sum(inter_prob_dist))
            da_indices[i] = np.random.choice(index_ls, size=1, p=inter_prob_dist_normd)
        father_chromes = survivors[da_indices]
        mother_chromes = survivors[ma_indices]
        return mother_chromes, father_chromes

    def mating_procedure(self, ma, da):
        # separate the ma and da chromosomes into respective control pulses
        ma_controls = np.hsplit(ma, self.N_controls)
        da_controls = np.hsplit(da, self.N_controls)
        num_parents = ma.shape[0]
        # Generate random numbers to probabilistically swap entire controls between
        # parent offspring
        swap_prob = np.random.normal(loc=0.5,scale=0.2,	size = self.N_controls)
        random_number = np.random.uniform(0,1,size = self.N_controls)
        # Loop over all possible control and probabilistically implement a swap
        off_1_ls = []
        off_2_ls = []
        off_1_ls.append(ma_controls[0])
        off_2_ls.append(da_controls[0])
        for idx in range(1,self.N_controls):
            if random_number[idx]>swap_prob[idx]:
                off_1_ls.append(da_controls[idx])
                off_2_ls.append(ma_controls[idx])
            else:
                off_1_ls.append(ma_controls[idx])
                off_2_ls.append(da_controls[idx])
        # Turn lists of jumbled controls into full arrays
        off_1 = np.concatenate(off_1_ls, axis=1)
        off_2 = np.concatenate(off_2_ls, axis=1)
        # define an array of combination parameters beta to use to combine the parent chromosomes
        # in a continuous way
        beta = np.random.uniform(0,1, size=(num_parents, self.N_var))
        beta_inverse = 1 - beta
        # Randomly select an array of ones to mask the beta array (this also randomly selects)
        # the indices to be swapped
        swap_array = np.random.randint(low=0,high=2,size=(num_parents, self.N_var))
        masked_beta = swap_array * beta
        masked_inverse_beta = swap_array * beta_inverse
        not_swap_array = np.mod((swap_array + 1),2)
        # Implement the b*O1 + (1-b)*02 on each chosen element
        new_off_1 = masked_beta * off_1 + masked_inverse_beta * off_2 + not_swap_array * off_1
        new_off_2 = masked_beta * off_2 + masked_inverse_beta * off_1 + not_swap_array * off_2
        offspring_array = np.concatenate((new_off_1,new_off_2),axis=0)
        return offspring_array


    def build_next_gen(self, survivors, offspring):
        # build next generation
        return np.concatenate((survivors, offspring), axis=0)


    def mutate(self, population, rate=0.3):
        # Mutate the new generation
        number_of_mutations = int((population.shape[0] - 1) * population.shape[-1] * rate)
        row_indices = np.random.choice(np.arange(1,int((population.shape[0]))), size=number_of_mutations)
        col_indices = np.random.choice(np.arange(0,int((population.shape[-1]))), size=number_of_mutations)
        mutated_population = np.copy(population)
        #mutated_population[row_indices, col_indices] = np.random.uniform(-1,1,size=number_of_mutations)
        mutated_population[row_indices, col_indices] = np.clip(population[row_indices, col_indices] + np.random.normal(loc=0, scale=0.2, size=number_of_mutations), a_min=-1, a_max=1)
        return mutated_population
