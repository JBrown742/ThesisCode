import numpy as np
import scipy as sci
import tensorflow as tf
import qutip as qt
import matplotlib.pyplot as plt
import os
import joblib
from ADDissipativeEngineering import ADDissipativeEngineering
from joblib import Parallel, delayed
import time
num_qubits =2
diss_span = 2
W4 = (qt.ket("0001") + qt.ket("0100") + qt.ket("0010") + qt.ket("1000"))/np.sqrt(4)
W3 = (qt.ket('001') + qt.ket('010') + qt.ket('100'))/np.sqrt(3)
W2 = (qt.ket('01') + qt.ket('10'))/np.sqrt(2)
GHZ2 = (qt.ket('00') + qt.ket('11'))/np.sqrt(2)
GHZ3 = (qt.ket('000') + qt.ket('111'))/np.sqrt(2)
r = 1/3
MEMS1 = (r/2)*(qt.projection(4,0,3) + qt.projection(4,3,0) + qt.projection(4,0,0) + qt.projection(4,3,3)) + (1-r)*qt.projection(4,1,1)
MEMS2 = (r/2)*(qt.projection(4,0,3) + qt.projection(4,3,0)) + (1/3)*(qt.projection(4,1,1)+ qt.projection(4,0,0) + qt.projection(4,3,3))
theta1 = np.pi
theta2 = np.pi
theta3 = np.pi
theta4 = np.pi
init_state = qt.ket('000')
plusX4 = qt.hadamard_transform(N=3)*init_state 
CP1 = qt.cphase(theta1, N=3, control=0, target=1)
CP2 = qt.cphase(theta2, N=3, control=1, target=2)
CP3 = qt.cphase(theta3, N=3, control=2, target=0)
# CP4 = qt.cphase(theta4, N=4, control=3, target=0)
cluster = (CP1 * CP2 * CP3 * plusX4)
# targ_state_pure  = W3
targ_state_name = 'MEMS2'
path = os.getcwd()+'/Reported_Results/'+targ_state_name+'/'+'N_{}_DissSpan_{}/'.format(num_qubits, diss_span)
if not os.path.isdir(path):
    os.makedirs(path)
batch_size = 1


AD = ADDissipativeEngineering(num_qubits=num_qubits, dissipator_span=diss_span, batchsize=batch_size, data="real")
"""
Optimization loop
"""
initial_learning_rate = 0.1
first_decay_steps =1000
# boundaries = [25* i for i in range(1,11)]
# values = [1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.000001]
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps,
    t_mul=1.5,
    alpha=0.0001,
    name=None
) 


# tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05,
#                                                               decay_steps=10,
#                                                               decay_rate=0.99)


# tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

# Later, whenever we perform an optimization step, we pass in the step.

optim_1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
targ_state_dm = MEMS2
sample_shape = tf.constant(np.real(qt.superoperator.operator_to_vector(targ_state_dm).full()), 
dtype=tf.float32).shape
targ_fl = tf.broadcast_to(tf.constant(qt.superoperator.operator_to_vector(targ_state_dm).full(), dtype=AD.dtype)[tf.newaxis,:], shape=(batch_size, *sample_shape))
size = AD.local_basis_size - 1
if AD.data=="real":
    shape = (batch_size, AD.N_dissipators, int((size*(size+1))/2))
    np_save_shape = batch_size*AD.N_dissipators*int((size*(size+1))/2)
else:
    shape = (batch_size, AD.N_dissipators, size, size)
    np_save_shape = batch_size*AD.N_dissipators*size*size
Diss_Vars = tf.Variable(np.random.uniform(-1,1,shape), dtype=tf.float64)
# H_Vars = tf.ones((batch_size, (2 * AD.N_qubits + AD.N_couple)))
n_local_params = AD.N_local_dissipative_params
cost_ls = []
steps = 0
checkpoint_count=0
checkpoint_steps=25
benchmark = 100000
benchmark_fidelity = 0.
max_steps = 10000
done=False
A_time = time.time()
while not done:
    print("Step: {}".format(steps))
    with tf.GradientTape() as tape:
        tape.watch(Diss_Vars)
        c_mat = AD.build_A(Diss_Vars)
        # H = AD.build_hamiltonian(H_Vars)
        Diss = AD.build_dissipator(c_mat)
        
        Liouv = Diss 
        out = tf.matmul(Liouv, targ_fl)
        dist = tf.reduce_sum(tf.abs(out))
        cost = dist
        cost_ls.append(cost)

    grads = tape.gradient(cost, [Diss_Vars])
    
    optim_1.apply_gradients(zip([grads[0]], [Diss_Vars]));
    steps+=1
    if cost<benchmark:
        Liouv_qt = qt.Qobj(Liouv[0].numpy(), type='super')
        rho_ss = qt.steadystate(Liouv_qt)
        # targ_dm = targ_state_pure * targ_state_pure.dag()
        targ_state = qt.Qobj(targ_state_dm.full())
        fid = qt.fidelity(targ_state, rho_ss)
        benchmark = cost
        if fid>benchmark_fidelity:
            benchmark_fidelity = fid
            best_Diss_Vars = Diss_Vars
            # best_H_Vars = H_Vars
            best_path = path+'/Best'
            if os.path.isdir(best_path)==False:
                os.makedirs(best_path)
            np.savetxt(best_path+'/Best_Diss_Vars', np.reshape(best_Diss_Vars.numpy(), newshape = np_save_shape))
            # np.savetxt(best_path+'/Best_Ham_Vars', np.reshape(best_H_Vars.numpy(), newshape = (batch_size * (2 * AD.N_qubits + AD.N_couple))))
            np.savetxt(best_path+'/Full_Cost', np.array(cost_ls))
    if steps%checkpoint_steps==0:
        B_time = time.time()
        time_taken_avg = (B_time - A_time)/checkpoint_steps
        A_time=time.time()
        print("\r Checkpont: {}, current best cost: {}, Current Best fidelity: {}".format(checkpoint_count, benchmark, benchmark_fidelity))   
        inter_path = path+'/checkpoint_{}'.format(checkpoint_count)
        if os.path.isdir(inter_path)==False:
            os.makedirs(inter_path)
        np.savetxt(best_path+'/Best_Diss_Vars', np.reshape(best_Diss_Vars.numpy(), newshape = np_save_shape))
        # np.savetxt(inter_path+'/Ham_Vars', np.reshape(H_Vars.numpy(), newshape = (batch_size * (2 * AD.N_qubits+AD.N_couple))))
        np.savetxt(inter_path+'/Cost', np.array(cost_ls))
        checkpoint_count+=1
    if steps>max_steps:
        done=True
    if benchmark_fidelity>0.9995:
        done = True
print("Best_cost = {}".format(benchmark))



    
