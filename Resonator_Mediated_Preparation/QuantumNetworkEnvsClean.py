
import numpy as np
import tensorflow as tf
import qutip as qt
from qutip import identity, basis, projection, tensor, create, destroy
import itertools
import matplotlib.pyplot as plt
import pickle

class QuanntumNetwork:

    def __init__(self, N, basis_dim, max_time=10, dicke_k=1, resolution=100, max_coupling=1, max_driving=2,
                 init_state_str="0",targ_state="Dicke", func_type="standard"):
        """
        attempt at coding n qubits in full generality
        """
        self.base_path = save_path
        
        self.N_controls =  N +1
        self.input_len = self.N_controls* basis_dim * 3
        self.basis_dim=basis_dim
        self.N_param_chunks= self.N_controls*3
        self.max_time = max_time
        self.resolution=resolution
        self.N=N
        self.func_type=func_type
        self.cav_dim=dicke_k+1
        self.qubit_names = ["qubit_{}".format(num) for num in range(self.N)]
        self.cav_freq = (2 * np.pi) * 5
        self.g = (2 * np.pi) * 200e-3
        self.kappa_0 = (2*np.pi) * 5e-6 
        self.gamma_1 = (2*np.pi) * 5e-3
        self.gamma_phi = (2*np.pi) * 0.31e-3
        self.max_coupling = max_coupling * self.g
        self.max_driving = max_driving * self.cav_freq
        self.times = np.linspace(0,max_time,resolution)
        self.exp_plus = np.exp(1j * self.cav_freq * self.times)
        self.exp_minus = np.exp(-1j * self.cav_freq * self.times)
                 
        """
        Initialize qubit operators accesible via dictionaries
        """

        self.op01 = {}
        self.op11 = {}
        self.sigz = {}
        for qubit_idx, qubit_name in enumerate(self.qubit_names):
            op01_ls = []
            op11_ls = []
            sigz_ls = []
            for i in range(len(self.qubit_names)):
                if i==qubit_idx:
                    op01_ls.append(projection(2,0,1))
                    op11_ls.append(projection(2,1,1))
                    sigz_ls.append(qt.sigmaz())
                else:
                    op01_ls.append(identity(2))
                    op11_ls.append(identity(2))
                    sigz_ls.append(identity(2))
            op01_ls.append(identity(self.cav_dim))
            op11_ls.append(identity(self.cav_dim))
            sigz_ls.append(identity(self.cav_dim))
            self.op01[qubit_name]=tensor(op01_ls)
            self.op11[qubit_name]=tensor(op11_ls)
            self.sigz[qubit_name]=tensor(sigz_ls)

        """
        initialize the creation op for the resonator
        """

        a_ls = [identity(2) for _ in range(self.N)]
        a_ls.append(destroy(self.cav_dim))
        self.a = tensor(a_ls)

        """
        Get the initial state
        """
        dims=[2]*self.N
        dims.append(self.cav_dim)
        if init_state_str=="0":
            string = "0"*(self.N+1)
            self.init_state = qt.ket2dm(qt.ket(string, dim=dims))
            
        else:
            self.init_state = qt.ket2dm(qt.ket(init_state_str, dim=dims))

        """
        get the list of strings corresponding to all strings of n bits with k ones
        and therefore build the target state
        """
        if targ_state=="Dicke":
            result = []
            for bits in itertools.combinations(range(self.N), dicke_k):
                s = ['0'] * self.N
                for bit in bits:
                    s[bit] = '1'
                result.append(''.join(s))  
            targ_pure_state = qt.tensor([qt.zero_ket(2) for i in range(self.N)])
            for string in result:
                targ_pure_state+=qt.ket(string)
            targ_state = (1/np.sqrt(len(result))) * targ_pure_state
            self.targ_state_ket = targ_state
            self.target_state_dm = targ_state * targ_state.dag()
        else:
            self.targ_state_ket = targ_state
            self.target_state_dm = qt.ket2dm(targ_state)
            
        ortho = qt.ket("0"*self.N) - self.targ_state_ket.dag()*qt.ket("0"*self.N)*self.targ_state_ket
        ls = [identity(2) for i in range(self.N)]
        ls.append(basis(self.cav_dim, self.cav_dim-1))
        self.fictitious_c_op = ortho*tensor(ls)
    
    def generic_func(self,amplitudes,phases, freqs):
        length = amplitudes.shape[0]
        func = np.zeros_like(self.times, dtype=np.complex64)
        for idx, t in enumerate(self.times):
            for i in range(length):
                func[idx] += amplitudes[i] * np.exp(1j*(freqs[i] * t + phases[i]))
#                 func[idx]+=amplitudes[i] * np.sin(freqs[i] * t + phases[i])      
        return func
    
    def generic_func_gauss(self, amps, mus, sigs):
        output = np.zeros_like(self.times)
        for i in range(amps.shape[0]):
            output+= amps[i] * np.exp(-(1/2)*(self.times - mus[i])**2/(sigs[i]**2))
        return output
    
    def get_params(self,chrome):
        coupling_amps = {}
        coupling_phases = {}
        coupling_freqs = {}
        driving_amps = {}
        driving_phases = {}
        driving_freqs = {}
        separated_chrome = np.hsplit(chrome, self.N_param_chunks)
        if self.func_type=="standard":
            for idx, name in enumerate(self.qubit_names):
                coupling_amps[name] = separated_chrome[3*idx]*self.max_coupling
                coupling_phases[name] = (separated_chrome[3*idx + 1] + 1) * (np.pi)
                coupling_freqs[name] = ((separated_chrome[3*idx + 2]*1) + 1.001)*(0.5) * ((2*np.pi)/self.times[-1])*6
            driving_amps = separated_chrome[-3]*self.max_driving
            driving_phases = (separated_chrome[-2] + 1) * (np.pi/2)
            driving_freqs = ((separated_chrome[-1]*1) + 1.001)*(0.5)*((2*np.pi)/self.times[-1])*6
        elif self.func_type=="gauss":
            for idx, name in enumerate(self.qubit_names):
                coupling_amps[name] = separated_chrome[3*idx]*self.max_coupling
                coupling_phases[name] = ((separated_chrome[3*idx + 1] + 1.2)/2.4) * self.times[-1]
                coupling_freqs[name] = (((separated_chrome[3*idx + 2]) + 1.24)/12)*self.times[-1]
            driving_amps = separated_chrome[3*idx]*self.max_driving
            driving_phases = ((separated_chrome[3*idx + 1] + 1)/2) * self.times[-1]
            driving_freqs = (((separated_chrome[3*idx + 2]) + 1.24)/12) *self.times[-1]
        return coupling_amps, coupling_phases, coupling_freqs, driving_amps, driving_phases, driving_freqs
    
    def decode_actions_func(self, chromosome):
        """
        The actions list will be of the form [delta_q_0, gamma_q_0, delta_q_1, gamma_q_1,.... etc]
        """
        coupling_pulses = {}
        coupling_amps, coupling_phases, coupling_freqs, \
        driving_amps, driving_phases, driving_freqs = self.get_params(chromosome)
        if self.func_type=="standard":
            for idx, name in enumerate(self.qubit_names):
                coupling = self.generic_func(coupling_amps[name], coupling_phases[name], coupling_freqs[name])
                coupling_pulses[name] = coupling
            driving_pulse = self.generic_func(driving_amps, driving_phases, driving_freqs)
        elif self.func_type=="gauss":
            for idx, name in enumerate(self.qubit_names):
                coupling = self.generic_func_gauss(coupling_amps[name], coupling_phases[name], coupling_freqs[name])
                coupling_pulses[name] = coupling
            driving_pulse = self.generic_func_gauss(driving_amps, driving_phases, driving_freqs)
        return coupling_pulses, driving_pulse


    
    def build_hamiltonian(self, couplings, cav_drive):
        H_full = []
        collapse_ops = []
        for name in self.qubit_names:
#             H_full.append([(self.op01[name]*self.a.dag() + self.op01[name].dag()*self.a), couplings[name]])
            H_full.append([self.op01[name]*self.a.dag(), couplings[name]])
            H_full.append([self.op01[name].dag()*self.a, np.conjugate(couplings[name])])
            H_full.append([self.a, cav_drive])
            H_full.append([self.a.dag(), np.conjugate(cav_drive)])
#             H_full.append([self.op01[name], self.exp_minus*drivings[name]])
#             H_full.append([self.op01[name].dag(), self.exp_plus*drivings[name]])
            collapse_ops.append(np.sqrt(self.gamma_1) * self.op01[name])
            collapse_ops.append(np.sqrt(self.gamma_phi/2) * self.sigz[name])
            collapse_ops.append(np.sqrt(self.kappa_0) * self.a)
    
        return H_full, collapse_ops
        
    

    def step(self, chromosome, his=True, Noise=True, state=None):
        if state==None:
            init_state=self.init_state
        else:
            init_state=state
            
        couplings, drive = self.decode_actions_func(chromosome)
        H, collapse = self.build_hamiltonian(couplings, drive)
        if Noise==False:
            collapse=[]
        if his==False:
            collapse.append()
            
            
        results = qt.mesolve(rho0=init_state, H = H, tlist=self.times, c_ops=collapse)
        state_histories = []
        fidelities = np.zeros_like(self.times)
        cav_state_his = []
        if his==True:
            for i in range(self.times.shape[0]):
                state_of_qubit_net = results.states[i].ptrace([j for j in range(self.N)])
                state_histories.append(state_of_qubit_net)
                state_of_cav = results.states[i].ptrace([self.N])
                cav_state_his.append(state_of_cav)
                fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
                fidelities[i]=fidelity
            self.info = {"state his": state_histories, "cav state his": cav_state_his, "fidelities":fidelities}
            reward=np.max(fidelities)
            
        else:
            state_of_qubit_net = results.states[-1].ptrace([j for j in range(self.N)])
            state_of_cav = results.states[-1].ptrace([self.N])
            self.info = {"state his": state_of_qubit_net, "cav state his": state_of_cav, "final state": results.states[-1]}
            fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
            reward=fidelity
        return reward
    
    
    
class QuantumNetworkPWC:

    def __init__(self, N_qubits, N_steps, init_state, targ_state, cav_dim, control='coupling', 
                        max_time=2, resolution=10, max_coupling=1, max_driving=1,
                        save_path=None, sharpness=2, qubit_detuning=0., max_detuning=2.):
        """
        attempt at coding n qubits in full generality
        """
        
                
        self.N_controls =  N_qubits + 1
        self.N_steps = N_steps
        self.input_len = self.N_controls*self.N_steps
        self.max_time = max_time
        self.resolution=resolution
        # Resolution must be EVEN!!!
     
        self.N=N_qubits
        # Forgive the nomenclature, dicke_k refers to number of excitations in a dicke state
        # the original states for  which this code was written
        self.cav_dim=cav_dim
        self.qubit_names = ["qubit_{}".format(num) for num in range(self.N)]
        self.cav_freq = (2 * np.pi) * 5
        self.g = (2 * np.pi) * 200e-3
        self.kappa_0 = (2*np.pi) * 5e-6
        self.gamma_1 = (2*np.pi) * 5e-3
        self.gamma_phi = (2*np.pi) * 0.31e-3
        self.control=control
        self.max_coupling = self.g * max_coupling
        self.max_detuning = max_detuning * (2*np.pi)
        self.qubit_detuning = qubit_detuning * self.g
        if control=='coupling':
            self.max_q_control = self.max_coupling
        if control=='detuning':
            self.max_q_control = self.max_detuning
        self.max_driving = max_driving * self.g
        self.times = np.linspace(0,max_time,resolution*N_steps)
        self.exp_plus = np.exp(1j * self.cav_freq * self.times)
        self.exp_minus = np.exp(-1j * self.cav_freq * self.times)
        self.sharpness=sharpness  
        """
        Initialize qubit operators accesible via dictionaries
        """

        self.op01 = {}
        self.op11 = {}
        self.sigz = {}
        for qubit_idx, qubit_name in enumerate(self.qubit_names):
            op01_ls = []
            op11_ls = []
            sigz_ls = []
            for i in range(len(self.qubit_names)):
                if i==qubit_idx:
                    op01_ls.append(projection(2,0,1))
                    op11_ls.append(projection(2,1,1))
                    sigz_ls.append(qt.sigmaz())
                else:
                    op01_ls.append(identity(2))
                    op11_ls.append(identity(2))
                    sigz_ls.append(identity(2))
            op01_ls.append(identity(self.cav_dim))
            op11_ls.append(identity(self.cav_dim))
            sigz_ls.append(identity(self.cav_dim))
            self.op01[qubit_name]=tensor(op01_ls)
            self.op11[qubit_name]=tensor(op11_ls)
            self.sigz[qubit_name]=tensor(sigz_ls)

        """
        initialize the creation op for the resonator
        """

        a_ls = [identity(2) for _ in range(self.N)]
        a_ls.append(destroy(self.cav_dim))
        self.a = tensor(a_ls)
        a_02_ls = [identity(2) for _ in range(self.N)]
        a_02_ls.append(projection(self.cav_dim, 0,2))
        self.a_02 = tensor(a_02_ls)
       
        self.init_state=init_state
        self.init_state_ket = qt.tensor(init_state, qt.ket('0', dim=[cav_dim]))
        self.init_state_dm = qt.ket2dm(self.init_state_ket)
        
        self.targ_state_ket = targ_state
        self.targ_state_dm = qt.ket2dm(targ_state)
        
        
        """
        Get Labels for plots etc...
        """        
        self.labels= []
        for k in range(2**self.N):
            self.labels.append(np.binary_repr(k,width=self.N))
            
        """
        Build a superposition of all qubit states
        """
            
        dims = [2 for _ in range(self.N)]
        dims.append(self.cav_dim)    
        self.qubit_super = qt.zero_ket((2**self.N)*self.cav_dim, dims=[dims,[1]*(self.N+1)])
        for string in self.labels:
            string+=str(self.cav_dim-1)
            self.qubit_super += qt.ket(string, dim=dims)
        self.qubit_super/=np.sqrt(len(self.labels))
        
        """
        Build a fictitious projector to discourage excited sate population
        """
        
        ortho = qt.ket("0"*self.N) - self.targ_state_ket.dag()*qt.ket("0"*self.N)*self.targ_state_ket
        orthog_full = tensor(ortho, basis(self.cav_dim, self.cav_dim-1))
        self.fictitious_c_op = np.sqrt(0.01*self.max_coupling)*orthog_full*self.qubit_super.dag()

    
    def tanh(self, offset, gap):
        ts=np.linspace(-1,1,self.resolution)
        error = (1 - (np.tanh(1*self.sharpness)))
        scale=1/(1-error)
        return (gap/2 * (scale * np.tanh(ts*self.sharpness) + 1)) + offset
    
    def get_pulses_from_pwc(self, pwc):
        full_pulse = np.zeros_like(self.times)
#         full_pulse[:int(self.resolution/2)]=[pwc[0]]*int(self.resolution/2)
        for i in range(0,(self.N_steps)):
            gap = pwc[i+1]-pwc[i]
            full_pulse[(i*self.resolution) : ((i+1)*self.resolution)] = self.tanh(offset=pwc[i],gap=gap)[:]
        # full_pulse[-int(self.resolution/2):]=[pwc[-1]]*int(self.resolution/2)
        return full_pulse
    
    
    def decode_actions_pwc(self, unscaled_chrome):
        """
        The actions list will be of the form [full_pulse_dq0, full_pulse_gammaq0....]
        """

        couplings = {}
        
        separated_chrome = np.hsplit(unscaled_chrome, self.N_controls)
        for idx, name in enumerate(self.qubit_names):
            if self.control=='detuning':
                coup = self.get_pulses_from_pwc(0.5*(separated_chrome[idx] + 1)*self.max_detuning)
            else:
                coup = self.get_pulses_from_pwc(separated_chrome[idx]*self.max_q_control)
            couplings[name] = coup
        drive = self.get_pulses_from_pwc(separated_chrome[-1]*self.max_driving)
        return couplings, drive

    
    def build_hamiltonian(self, qubit_controls, drive):
        H_full = []
        collapse_ops = []
        for name in self.qubit_names:
            if self.control=='detuning':
                H_full.append([self.op11[name], qubit_controls[name]])
                H_full.append((self.op01[name]*self.a.dag() + self.op01[name].dag()*self.a) * self.max_coupling)
            elif self.control=='coupling':
                H_full.append(self.op11[name]*self.qubit_detuning)
                H_full.append([(self.op01[name]*self.a.dag() + self.op01[name].dag()*self.a), qubit_controls[name]])
            collapse_ops.append(np.sqrt(self.gamma_1) * self.op01[name])
            collapse_ops.append(np.sqrt(self.gamma_phi/2) * self.sigz[name])
        collapse_ops.append(np.sqrt(self.kappa_0) * self.a)
        H_full.append([(self.a + self.a.dag()), drive])
        
        return H_full, collapse_ops
        
    

    def step(self, chromosome, his=False, Noise=True):
        
        init_state=self.init_state_ket
        init_state_dm=self.init_state_dm
        qubit_control, drive = self.decode_actions_pwc(chromosome)
        # Can only be run after plot_initializer
        H, collapse = self.build_hamiltonian(qubit_control, drive)
        if Noise==False and his==True:
            results = qt.sesolve(psi0=init_state, H = H, tlist=self.times)
        elif Noise==False and his==False:
            collapse=[]
            collapse.append(self.fictitious_c_op)
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        elif Noise==True and his==False:
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        elif Noise==True and his==True:
            collapse.append(self.fictitious_c_op)
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        
        state_histories = []
        fidelities = np.zeros_like(self.times)
        cavity_excited_pops = np.zeros_like(self.times)
        cav_state_his = []
        full_state_his = []
        if his==True:
            for i in range(self.times.shape[0]):
                if Noise==False and his==True:
                    inter_state = qt.ket2dm(results.states[i])
                else:
                    inter_state = results.states[i]
                full_state_his.append(inter_state)
                state_of_qubit_net = inter_state.ptrace([j for j in range(self.N)])
                state_histories.append(state_of_qubit_net)
                state_of_cav = inter_state.ptrace([self.N])
                cavity_excited_pops[i] = state_of_cav.diag()[-1]
                cav_state_his.append(state_of_cav)
                fidelity = qt.fidelity(state_of_qubit_net, self.targ_state_dm)
                fidelities[i]=fidelity
            self.info = {"state his": state_histories, "cav state his": cav_state_his, "full state his":full_state_his, "fidelities":fidelities}
            max_fid = np.max(fidelities)
            max_index=max(1,np.argmax(fidelities))
            if fidelities[max_index:].shape[0]> self.resolution*4:
                reward= np.round((max_fid 
                                + 0.5 * np.sum(fidelities[max_index:(max_index + self.resolution*4)])/fidelities[max_index:(max_index + self.resolution*4)].shape[0] 
                                - 0.1 * (np.sum(cavity_excited_pops)/self.times.shape[0])), decimals=5)
            else:
                reward=np.round((max_fid 
                                + 0.5 * np.sum(fidelities[max_index:])/fidelities[max_index:].shape[0] 
                                - 0.1 * (np.sum(cavity_excited_pops)/self.times.shape[0])), decimals=5)
        else:
            state_of_qubit_net = results.states[-1].ptrace([j for j in range(self.N)])
            state_of_cav = results.states[-1].ptrace([self.N])
            self.info = {"state his": state_of_qubit_net, "cav state his": state_of_cav, "final state": results.states[-1]}
            fidelity = qt.fidelity(state_of_qubit_net, self.targ_state_dm)
            final_control_vals = np.zeros(self.N_controls)
            for i, name in enumerate(self.qubit_names):
                final_control_vals[i]= qubit_control[name][-1]
            final_control_vals[-1] = drive[-1]
            reward=np.round(1 + fidelity - 0.2 * (np.sum(np.abs(final_control_vals))/4), decimals=5)
        return reward


    def run_dynamics(self, qubit_control, drive, Noise=False):
        init_state=self.init_state_ket
        init_state_dm=self.init_state_dm
        # Can only be run after plot_initializer
        H, collapse = self.build_hamiltonian(qubit_control, drive)
        if Noise==False and his==True:
            results = qt.sesolve(psi0=init_state, H = H, tlist=self.times)
        elif Noise==False and his==False:
            collapse=[]
            collapse.append(self.fictitious_c_op)
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        elif Noise==True and his==False:
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        elif Noise==True and his==True:
            collapse.append(self.fictitious_c_op)
            results = qt.mesolve(rho0=init_state_dm, H = H, tlist=self.times, c_ops=collapse)
        
        state_histories = []
        fidelities = np.zeros_like(self.times)
        cavity_excited_pops = np.zeros_like(self.times)
        cav_state_his = []
        full_state_his = []
        for i in range(self.times.shape[0]):
            if Noise==False and his==True:
                inter_state = qt.ket2dm(results.states[i])
            else:
                inter_state = results.states[i]
            full_state_his.append(inter_state)
            state_of_qubit_net = inter_state.ptrace([j for j in range(self.N)])
            state_histories.append(state_of_qubit_net)
            # state_of_cav = inter_state.ptrace([self.N])
            cavity_excited_pops[i] = state_of_cav.diag()[-1]
            cav_state_his.append(state_of_cav)
            fidelity = qt.fidelity(state_of_qubit_net, self.targ_state_dm)
            fidelities[i]=fidelity
            self.info = {"state his": state_histories, "cav state his": cav_state_his, "full state his":full_state_his, "fidelities":fidelities}
        return np.max(fidelities)

    def save(self, chromosome):
        # ONLY RUN AFTER PLOT INITIALIZER 
        with open(self.path+'/env', 'wb') as pickle_file:
            pickle.dump(self,  pickle_file)
        np.savetxt(self.path+'/chromosome', chromosome)

    def plot_initializer(self, chromosome, Noise=False, alt_path_ext=None, controls=None, 
                        noise_factors=[1,1,1]):
        # alternate path extension must be a string and must correspond to a subdirectory 
        # that has been created within self.base_path directory, mainly for checkpointing!
        if alt_path_ext==None:
            self.path=self.base_path
        else:
            self.path=self.base_path+alt_path_ext
        self.kappa_0 = noise_factors[0] * (2*np.pi) * 5e-6
        self.gamma_1 = noise_factors[1] * (2*np.pi) * 5e-3
        self.gamma_phi = noise_factors[2] * (2*np.pi) * 0.31e-3
        self.chromosome=chromosome
        self.noise=Noise
        if controls==None:
            res_ = self.step(chromosome, his=True, Noise=Noise)
        else:
            res_ = self.run_dynamics(controls[0],controls[1], Noise=Noise)
        self.fid = self.info["fidelities"]
        self.index = self.times.shape[0]-1
        self.best_index = np.argmax(self.fid)
        self.state_his=self.info["state his"]
        self.best_state = self.state_his[self.best_index]
        self.cav_his = self.info["cav state his"]
        return 
    
    def plot_fidelity(self, save=False, show=False):
        plt.figure(1, figsize=(10,6))
        plt.plot(self.times[:self.index], self.fid[:self.index],linewidth=2)
        plt.title("State Fidelity during dynamics of protocol\n Max Fidelity = {}".format(np.round(self.fid[self.best_index], decimals=4)), fontsize=18)
        plt.xlabel(r"Time/ns", fontsize=16)
        plt.ylabel(r"fidelity", fontsize=16)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Fidelity_Plot", dpi=300, layout='tight')
        else:
            pass
        return
    
    def plot_control_pulses(self, save=False, show=False):
        # Run dynamics to get best index
        coups, drive = self.decode_actions_pwc(self.chromosome)
        names = self.qubit_names
        title_label = self.control
        plt.figure(2)
        n_rows=int(np.ceil(self.N_controls/2))
        fig, axes = plt.subplots(n_rows,2, figsize=(18,11), sharey=False, sharex=True)
        for idx, name in enumerate(names):
            axes[idx//2, idx%2].plot(self.times[:self.index], coups[name][:self.index]/self.g, linewidth=2)
            axes[idx//2, idx%2].set_title("{} {}".format(name, title_label), fontsize=20)
            axes[idx//2, idx%2].set_ylabel(r"${} /g_0$".format(title_label), fontsize=20)
            axes[idx//2, idx%2].tick_params(labelsize=12, width=1.5, length=3)
        axes[(idx+1)//2,(idx+1)%2].plot(self.times[:self.index], (drive[:self.index])/(self.max_driving*self.g), linewidth=2)
        axes[(idx+1)//2,(idx+1)%2].set_title("cavity driving", fontsize=20)
        axes[(idx+1)//2,(idx+1)%2].set_ylabel(r"$\xi  /{} g_0$".format(np.round(self.max_driving/self.g, decimals=2)), fontsize=20)
        axes[(idx+1)//2,(idx+1)%2].tick_params(labelsize=12, width=1.5, length=3)
        axes[-1,0].set_xlabel("Time/ns", fontsize=20)
        axes[(idx+1)//2,(idx+1)%2].set_xlabel("Time/ns", fontsize=20)
        
        if (idx+2)%2==1:
            axes[(idx+2)//2,(idx+2)%2].axis('off')
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Control_Plots", dpi=600, layout='tight')
        else:
            pass
        return
    

    
    def plot_histograms(self, save=False, show=False):
        xlabels = self.labels
        targ_array=self.targ_state_dm.full()
        targ_max = np.max(np.abs(targ_array))
        plt.figure(3)
        fig1, ax1 = qt.visualization.matrix_histogram(self.targ_state_dm, xlabels, xlabels, 
                                                    limits=[-targ_max, targ_max],
                                                    title="target state density matrix")
        # ax1.autoscale()
        ax1.set_title("Target state density matrix", fontsize=20)
        ax1.view_init(azim=-55, elev=35)
        if save==True:
            plt.savefig(self.path+"/Target_state_historgram", dpi=600)
        else:
            pass
            
        plt.figure(4)
        best_array=self.best_state.full()
        best_max = np.max(np.abs(best_array))
        fig2, ax2 = qt.visualization.matrix_histogram(self.best_state, xlabels, xlabels,
                                                    limits=[-best_max, best_max],
                                                   title="Final state Density Matrix")
        # ax2.autoscale()
        ax2.set_title("Final state density matrix", fontsize=20)
        ax2.view_init(azim=-55, elev=35)
        if save==True:
            plt.savefig(self.path+"/Best_state_historgram", dpi=600)
        else:
            pass
        if show==True:
            plt.show()
        return

    def plot_cavity(self, save=False, show=False):
        cav_populations = np.zeros((self.cav_dim, self.times.shape[0]))
        for idx, t in enumerate(self.times):
            state_diag = self.cav_his[idx].diag()
            cav_populations[:,idx] = state_diag[:]
        plt.figure(500, figsize = (12,8))
        for i in range(self.cav_dim):
            plt.plot(self.times[:self.index], cav_populations[i,:self.index], label = r"$|{}>$".format(i), linewidth=2)
        plt.legend()
        plt.xlabel(r"Time/ns", fontsize=16)
        plt.ylabel(r"Population", fontsize=16)
        plt.title("Populations for cavity state during dynamics", fontsize=18)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Cavity_Population_dynamics", dpi=300)
        else:
            pass
        return
    

    def plot_qubits(self, save=False, show=False):
        population = np.zeros((self.N,self.times.shape[0]))
        for idx, t in enumerate(self.times):
            for n in range(self.N):
                state = self.state_his[idx].ptrace([n])
                diag_state = state.diag()
                population[n,idx]=diag_state[1]
        plt.figure(600, figsize=(10,6))
        for idx,name in enumerate(self.qubit_names):
            plt.plot(self.times[:self.index], population[idx,:self.index], label=r"$\mathcal{Q}$ "+str(idx), linewidth=2)
        plt.legend(fontsize = 12)
        plt.title(r"$<1|\rho_{\mathcal{Q}}|1>$, of reduced state for each qubit.", fontsize=18)
        plt.xlabel(r"Time/$ns$", fontsize=16)
        plt.ylabel(r"$<1|\rho_{Q}|1>$", fontsize=16)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Qubit_Population_dynamics", dpi=300)
        else:
            pass
        return
 
 
#____________________________________________________________________________________________________
#----------------------------------------------------------------------------------------------------       
        
class QuantumNetworkPWCQubitDriven:

    def __init__(self, N_qubits, N_steps, max_time=10, dicke_k=1, resolution=10, max_coupling=1, max_driving=0.01,
                 init_state_str="0",targ_state="Dicke", save_path=None):
        """
        attempt at coding n qubits in full generality
        """
        self.path = save_path        
        self.N_controls =  2*N_qubits
        self.N_steps = N_steps
        self.input_len = self.N_controls*self.N_steps
        self.max_time = max_time
        self.resolution=resolution
        self.N=N_qubits
        self.cav_dim=dicke_k+1
        self.qubit_names = ["qubit_{}".format(num) for num in range(self.N)]
        self.cav_freq = (2 * np.pi) * 5
        self.g = (2 * np.pi) * 200e-3
        self.kappa_0 = (2*np.pi) * 5e-6
        self.gamma_1 = (2*np.pi) * 5e-3
        self.gamma_phi = (2*np.pi) * 0.31e-3
        self.max_coupling = max_coupling * self.g
        self.max_driving = max_driving * self.g
        self.times = np.linspace(0,max_time,resolution*N_steps)
        self.exp_plus = np.exp(1j * self.cav_freq * self.times)
        self.exp_minus = np.exp(-1j * self.cav_freq * self.times)
                 
        """
        Initialize qubit operators accesible via dictionaries
        """

        self.op01 = {}
        self.op11 = {}
        self.sigz = {}
        for qubit_idx, qubit_name in enumerate(self.qubit_names):
            op01_ls = []
            op11_ls = []
            sigz_ls = []
            for i in range(len(self.qubit_names)):
                if i==qubit_idx:
                    op01_ls.append(projection(2,0,1))
                    op11_ls.append(projection(2,1,1))
                    sigz_ls.append(qt.sigmaz())
                else:
                    op01_ls.append(identity(2))
                    op11_ls.append(identity(2))
                    sigz_ls.append(identity(2))
            op01_ls.append(identity(self.cav_dim))
            op11_ls.append(identity(self.cav_dim))
            sigz_ls.append(identity(self.cav_dim))
            self.op01[qubit_name]=tensor(op01_ls)
            self.op11[qubit_name]=tensor(op11_ls)
            self.sigz[qubit_name]=tensor(sigz_ls)

        """
        initialize the creation op for the resonator
        """

        a_ls = [identity(2) for _ in range(self.N)]
        a_ls.append(destroy(self.cav_dim))
        self.a = tensor(a_ls)

        """
        Get the initial state
        """
        dims=[2]*self.N
        dims.append(self.cav_dim)
        if init_state_str=="0":
            string = "0"*(self.N+1)
            state_pure = qt.ket(string, dim=dims)
            self.init_state = qt.ket2dm(state_pure)
            
        else:
            self.init_state = qt.ket2dm(qt.ket(init_state_str, dim=dims))

        """
        get the list of strings corresponding to all strings of n bits with k ones
        and therefore build the target state
        """
        if targ_state=="Dicke":
            result = []
            for bits in itertools.combinations(range(self.N), dicke_k):
                s = ['0'] * self.N
                for bit in bits:
                    s[bit] = '1'
                result.append(''.join(s))  
            targ_pure_state = qt.tensor([qt.zero_ket(2) for i in range(self.N)])
            for string in result:
                targ_pure_state+=qt.ket(string)
            targ_state = (1/np.sqrt(len(result))) * targ_pure_state
            self.target_state_dm = targ_state * targ_state.dag()
        else:
            self.target_state_dm = targ_state

    
    def tanh(self, offset, gap, sharpness=2):
        ts=np.linspace(-1,1,self.resolution)
        return ((gap/2) * (np.tanh(ts*sharpness)+1)) + offset
    
    def get_pulses_from_pwc(self, pwc):
        full_pulse = np.zeros_like(self.times)
        full_pulse[:int(self.resolution/2)]=[pwc[0]]*int(self.resolution/2)
        for i in range(0,(self.N_steps-1)):
            gap = pwc[i+1]-pwc[i]
            full_pulse[(i*self.resolution) + int(self.resolution/2) : ((i+1)*self.resolution) + int(self.resolution/2)] = self.tanh(offset=pwc[i],gap=gap)[:]
        full_pulse[-int(self.resolution/2):]=[pwc[-1]]*int(self.resolution/2)
        return full_pulse
    
    def decode_actions_pwc(self, unscaled_chrome):
        """
        The actions list will be of the form [full_pulse_dq0, full_pulse_gammaq0....]
        """

        couplings = {}
        drivings = {}
        separated_chrome = np.hsplit(unscaled_chrome, self.N_controls)
        for idx, name in enumerate(self.qubit_names):
            coup = self.get_pulses_from_pwc(separated_chrome[2*idx]*self.max_coupling)
            #coupling_im = self.get_pulses_from_pwc(separated_chrome[2*idx+1]*self.max_coupling)
            couplings[name] = coup
            #couplings[name] = left_over[idx]
            drive = self.get_pulses_from_pwc(separated_chrome[2*idx + 1]*self.max_driving)
            drivings[name] = drive
        return couplings, drivings

    
    def build_hamiltonian(self, couplings, drivings):
        H_full = []
        collapse_ops = []
        for name in self.qubit_names:
#             H_full.append([self.op11[name], qubit_deltas[name]])
#             H_full.append((self.op01[name]*self.a.dag() + self.op01[name].dag()*self.a) * self.max_coupling)
            H_full.append([self.op01[name]*self.a.dag(), couplings[name]])
            H_full.append([self.op01[name].dag()*self.a, np.conjugate(couplings[name])])
            H_full.append([self.op01[name], self.exp_minus*drivings[name]])
            H_full.append([self.op01[name].dag(), self.exp_plus*drivings[name]])
            collapse_ops.append(np.sqrt(self.gamma_1) * self.op01[name])
            collapse_ops.append(np.sqrt(self.gamma_phi/2) * self.sigz[name])
        collapse_ops.append(np.sqrt(self.kappa_0) * self.a)
#         H_full.append([(self.a + self.a.dag()), drive])
        return H_full, collapse_ops
        
    

    def step(self, chromosome, his=False, Noise=True, state=None):
        if state==None:
            init_state=self.init_state
        else:
            init_state=state
            
        couplings, drivings = self.decode_actions_pwc(chromosome)
        H, collapse = self.build_hamiltonian(couplings, drivings)
        if Noise==True:
            results = qt.mesolve(rho0=init_state, H = H, tlist=self.times, c_ops=collapse)
        else:
            results = qt.mesolve(rho0=init_state, H = H, tlist=self.times)
        state_histories = []
        fidelities = np.zeros_like(self.times)
        cav_state_his = []
        highest_excited = np.zeros_like(self.times)
        if his==True:
            for i in range(self.times.shape[0]):
                state_of_qubit_net = results.states[i].ptrace([j for j in range(self.N)])
                state_histories.append(state_of_qubit_net)
                state_of_cav = results.states[i].ptrace([self.N])
                highest_excited[i] = state_of_cav.diag()[-1]
                cav_state_his.append(state_of_cav)
                fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
                fidelities[i]=fidelity
            self.info = {"state his": state_histories, "cav state his": cav_state_his, "fidelities":fidelities}
            reward=np.max(fidelities)
            
        else:
            state_of_qubit_net = results.states[-1].ptrace([j for j in range(self.N)])
            state_of_cav = results.states[-1].ptrace([self.N])
            self.info = {"state his": state_of_qubit_net, "cav state his": state_of_cav, "final state": results.states[-1]}
            fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
            reward=fidelity
        return reward

    def save(self, chromosome):
        with open(self.path+'/env', 'wb') as pickle_file:
            pickle.dump(self,  pickle_file)
        np.savetxt(self.path+'/chromosome', chromosome)
    
    
    def plot_initializer(self, chromosome, Noise=False):
        self.chromosome=chromosome
        self.noise = Noise
    
        res_ = self.step(chromosome, his=True, Noise=Noise)
        self.fid = self.info["fidelities"]
        self.index = self.fid.shape[-1] - 1  
        self.state_his=self.info["state his"]
        self.cav_his = self.info["cav state his"]
        return 
    
    def plot_fidelity(self, save=False, show=False):
        plt.figure(1, figsize=(9,6))
        plt.plot(self.times[:self.index], self.fid[:self.index],linewidth=2)
        plt.title("State Fidelity during dynamics of protocol\n Max Fidelity = {}".format(np.round(self.fid[self.index], decimals=4)), fontsize=18)
        plt.xlabel(r"Time/ns", fontsize=16)
        plt.ylabel(r"fidelity", fontsize=16)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Fidelity_Plot", dpi=600, layout='tight')
        else:
            pass
        plt.clf()
        return
    
    def plot_control_pulses(self, save=False, show=False):
        # Run dynamics to get best index
        coups, drives = self.decode_actions_pwc(self.chromosome)
        names = self.qubit_names
        plt.figure(6)
        fig, axes = plt.subplots(2,3, figsize=(27,11), sharey=True, sharex=True)
        for idx, name in enumerate(names):
            axes[0, idx%3].plot(self.times[:self.index], coups[name][:self.index]/self.g, linewidth=2)
            axes[0, idx%3].set_title("{} coupling ".format(name), fontsize=20)
            axes[0, idx%3].set_ylabel(r"$G /g_0$", fontsize=20)
            axes[0, idx%3].tick_params(labelsize=12, width=1.5, length=3)
            axes[1, idx%3].plot(self.times[:self.index], (drives[name][:self.index])/(self.max_driving*self.g), linewidth=2)
            axes[1, idx%3].set_title("cavity driving", fontsize=20)
            axes[1, idx%3].set_ylabel(r"$\xi  /{} g_0$".format(np.round(self.max_driving/self.g, decimals=2)), fontsize=20)
            axes[1, idx%3].tick_params(labelsize=12, width=1.5, length=3)
        axes[-1,0].set_xlabel("Time/ns", fontsize=20)
        axes[-1,1].set_xlabel("Time/ns", fontsize=20)
        axes[-1,2].set_xlabel("Time/ns", fontsize=20)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Control_Plots", dpi=600, layout='tight')
        else:
            pass
        plt.clf()
        return
    
    def get_labels(self):
        result = []
        for k in range(2**self.N):
            result.append(np.binary_repr(k,width=self.N))
        return result
    
    def plot_histograms(self, save=False, show=False):
        xlabels = self.get_labels()

        best_state = self.state_his[self.index]
        plt.figure(3)
        fig1, ax1 = qt.visualization.matrix_histogram(self.target_state_dm, xlabels, xlabels, limits=[-1/2,1/2],
                                                   title="target state density matrix")
        ax1.autoscale()
        ax1.set_title("Target state density matrix", fontsize=20)
        ax1.view_init(azim=-55, elev=35)
        if save==True:
            plt.savefig(self.path+"/Target_state_historgram", dpi=600)
        else:
            pass
            
        plt.clf()
        plt.figure(4)
        fig2, ax2 = qt.visualization.matrix_histogram(best_state, xlabels, xlabels, limits=[-1/2,1/2],
                                                   title="Final state Density Matrix")
        ax2.autoscale()
        ax2.set_title("Final state density matrix", fontsize=20)
        ax2.view_init(azim=-55, elev=35)
        if save==True:
            plt.savefig(self.path+"/Best_state_historgram", dpi=600)
        else:
            pass
        if show==True:
            plt.show()
 #        plt.clf()
        return

    def plot_cavity(self, save=False, show=False):
        cav_populations = np.zeros((self.cav_dim, self.times.shape[0]))
        for idx, t in enumerate(self.times):
            state_diag = self.cav_his[idx].diag()
            cav_populations[:,idx] = state_diag[:]
        plt.figure(5, figsize = (9,6))
        for i in range(self.cav_dim):
            plt.plot(self.times[:self.index], cav_populations[i,:self.index], label = r"$|{}>$".format(i), linewidth=2)
        plt.legend()
        plt.xlabel(r"Time/ns", fontsize=16)
        plt.ylabel(r"Population", fontsize=16)
        plt.title("Populations for cavity state during dynamics", fontsize=18)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Cavity_Population_dynamics", dpi=300)
        else:
            pass
        plt.clf()
        return
    

    def plot_qubits(self, save=False, show=False):
        population = np.zeros((self.N,self.times.shape[0]))
        for idx, t in enumerate(self.times):
            for n in range(self.N):
                state = self.state_his[idx].ptrace([n])
                diag_state = state.diag()
                population[n,idx]=diag_state[1]
        plt.figure(6, figsize=(9,6))
        for idx,name in enumerate(self.qubit_names):
            plt.plot(self.times[:self.index], population[idx,:self.index], label=r"$\mathcal{Q}$ "+str(idx), linewidth=2)
        plt.legend(fontsize = 12)
        plt.title(r"$<1|\rho_{\mathcal{Q}}|1>$, of reduced state for each qubit.", fontsize=18)
        plt.xlabel(r"Time/$ns$", fontsize=16)
        plt.ylabel(r"$<1|\rho_{Q}|1>$", fontsize=16)
        if show==True:
            plt.show()
        if save==True:
            plt.savefig(self.path+"/Qubit_Population_dynamics", dpi=300)
        else:
            pass
        plt.clf()
        return