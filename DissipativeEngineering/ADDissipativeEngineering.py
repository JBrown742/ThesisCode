import numpy as np
import scipy as sci
import tensorflow as tf
import tensorflow_probability as tfp
import qutip as qt
import math
import itertools

class ADDissipativeEngineering:

    def __init__(self, num_qubits, dissipator_span, omega_max=5, g_percent=5, batchsize=1, data="real"):
        if data=="real":
            self.dtype=tf.float32
        elif data=="complex":
            self.dtype=tf.complex64
        self.data=data
        self.N_qubits = num_qubits
        self.dim_global_H = 2 ** num_qubits
        self.dim_local_H = 2 ** dissipator_span
        self.global_basis_size = self.dim_global_H ** 2
        # Calculate the number of basis matrices for the space 4 ** num subsystems in each dissipator
        self.local_basis_size = self.dim_local_H ** 2
        # Assume the span of each dissipator is the same
        self.diss_span = dissipator_span
        self.omega_max = 2 * np.pi * omega_max  #GHz
        self.g_max = 2 * np.pi * omega_max * (g_percent/100)
        # Calculate the number of coupling for a fully connected network of
        # qubits (N * (N-1))/2. (Factor of two accounts for double counts)
        self.N_couple = int((num_qubits * (num_qubits - 1))/2)
        # Calculate the number of baths based on the bath size
        self.N_dissipators = int((math.factorial(num_qubits)/(math.factorial(dissipator_span) * math.factorial((num_qubits-dissipator_span)))))
        # Calculate the number of parameters in the dissipator c_matrix for each dissipator being ((dim_H_local ** 2 -1), (dim_H_local ** 2 -1))
        self.N_local_dissipative_params = int((self.local_basis_size - 1) ** 2)
        # Calculate total num dissipative params
        self.N_total_dissipative_params = int(self.N_local_dissipative_params * self.N_dissipators)
        self.N_total_hamiltonian_params = int(2 * self.N_qubits + self.N_couple)
        # Calculate the number of variables for the system as N_var = num_qubit * frequencies + N_couple * couplings + N_dissipator_params
        self.N_var = int(2 * self.N_qubits + self.N_couple + self.N_total_dissipative_params)
        self.diss_indices = list(itertools.combinations(np.arange(num_qubits), dissipator_span))
        self.coup_indices = list(itertools.combinations(np.arange(num_qubits), 2))
        self.batchsize=batchsize
        self.gamma = tf.constant(2 * np.pi * 5e-3, dtype=tf.complex64)
        self.gamma_phi = tf.constant(2 * np.pi * 0.31e-3, dtype=tf.complex64)
        self.max_val = 2 * (self.local_basis_size - 1)
        """
        Define the local qubit creation and annihilation operators
        """      
        # Define the sigma plus and minus operators for each local qubit in standard representation
        self.sig_plus_ls = []
        self.sig_minus_ls = []
        original_ls = [qt.identity(2)] * self.N_qubits
        for idx in range(self.N_qubits):
            sig_m_ls = original_ls.copy()
            sig_p_ls = original_ls.copy()
            sig_m_ls[idx] = qt.projection(2,0,1)
            sig_p_ls[idx] = qt.projection(2,1,0)
            self.sig_plus_ls.append(qt.tensor(sig_p_ls))
            self.sig_minus_ls.append(qt.tensor(sig_m_ls))
            
        
        # define the hamiltonian dimension
        self.ham_dims = self.sig_plus_ls[0].dims 
        """
        BASIS BUILDING CODE: Builds the basis of 'Generalized Paulis' pertaining to the specific architecture.
        """
        # Define a list of the pauli based spin operators to act as a trace orthonormal matrix basis for a single qubit
        paulis = [qt.identity(2) * (1/np.sqrt(2)), qt.sigmax() * (1/np.sqrt(2)),      
                -1j * qt.sigmay() * (1/np.sqrt(2)), qt.sigmaz() * (1/np.sqrt(2))]
        pauli_str_labels = [r"$I$", r"$X$", r"$Y$", r"$Z$"]         
        """
         1. Build a list containing each element of the basis for a single dissipator that spans 
            self.diss_span qubits, each element of the list will itself be a list so the full list will 
            look like,
                        [[sigma_0, ..., sigma_0], 
                                   ... , 
                         [sigma_3, ..., sigma_3]].
            This is because, in general the dissipator and the full hilbert space will have different dimensions 
            so we need to filter identities into the relevant place in these list elements before we take the
            tensor product. Eg. A dissipator the spans qubits 1 and 3 in a 2 qubit system, the list basis elements 
            will be [sigma_0, sigma_0] but the actual basis vector embedded in the larger hilbert space is
            tensor([sigma_0, Identity, sigma_0])
        """          
        # Initialize an empty list to store the sets of basis matrices for each dissipator
        self.local_basis_ls = []
        self.index_ls = []
        self.label_ls = []
        for i in range(self.local_basis_size):
            pauli_ls = np.zeros(self.diss_span, dtype=object)
            inner_idxs = []
            inner_labels = []
            for k, j in enumerate(reversed(range(self.diss_span))):
                idx = (i%(4**(j+1)))//(4**j)
                inner_idxs.append(idx)
                inner_labels.append(pauli_str_labels[idx])
                # Turn basis index (Turn 1,2,3,...,N*2) into (0,0,...,0), (0,0,...,1),..., (3,3,...,3)
                # if location = (N **2,...,2,1,0) then index_val = index % (4 ** (loc + 1))// 4 ** loc
                pauli_ls[k] = paulis[idx]
                # tensor product the basis matrices and append to local dissipator basis list
            self.index_ls.append(inner_idxs)
            self.label_ls.append(r''.join(inner_labels))
            self.local_basis_ls.append(pauli_ls)
        """
        2. Now take the local pauli basis list for the dissipator and build N_dissipator dissipators
           using the relevant indices for each dissipator and filling the other entries with
           the identity as per 1.
        """
        # Now build the list of basis for each Dissipator
        self.full_basis_ls = []
        self.full_basis_labels = []
        # loop over each different dissipator
        for diss_idx in range(self.N_dissipators):
            # get the indices of the qubits over which this dissipator acts
            diss_loc_idxs = self.diss_indices[diss_idx]
            local_basis = []
            local_label = []
            for k in range(self.local_basis_size):
                # within each dissipator, loop over the basis size!
                basis_elem = [qt.identity(2)] * self.N_qubits
                label_elem = [r" $I$ "] * self.N_qubits
                for i, idx in enumerate(diss_loc_idxs):
                    # install the local basis matrix S_k,i in locations idx
                    basis_elem[idx] = self.local_basis_ls[k][i]
                local_basis.append(qt.tensor(basis_elem))
            self.full_basis_ls.append(local_basis)
            # Results in a list of lists enumerated as [Dissipator index, basis index]

    
        """
        3. Now we take the full_basis_ls and write each element in Fock-Liouville space
           (Choi-Isomorphism). Here we will have a matrix where the (i,j)-th element is
           the FL representation of the general Lindblad Dissipator written with operators
           F_i and F_j.
        """
        self.sample_shape = qt.spre(self.full_basis_ls[0][0]).full().shape
        size_of_things = len(self.full_basis_ls[diss_idx][1:])
        self.FL_basis_full = np.zeros((self.batchsize, self.N_dissipators, size_of_things,size_of_things,*self.sample_shape), 
                                        dtype=np.float32)
        # loop over each different dissipator
        for diss_idx in range(self.N_dissipators):
            # get the indices of the qubits over which this dissipator acts
            for i, basis_i in enumerate(self.full_basis_ls[diss_idx][1:]):
                for j, basis_j in enumerate(self.full_basis_ls[diss_idx][1:]):
                    fl_op = (qt.sprepost(basis_i, basis_j.dag())
                                - (1/2) * (qt.spre((basis_j.dag() * basis_i)) +
                                qt.spost((basis_j.dag() * basis_i))))
                    self.FL_basis_full[:,diss_idx, i,j] = tf.broadcast_to(tf.constant(fl_op.full()[tf.newaxis,:,:],
                                                                                  dtype=tf.float32),
                                                                                  shape=(self.batchsize,*self.sample_shape))
    
        """
        4. Same as above. Cast the qubit creation and annihilation operators into FL rep.
        """

        self.sig_p_pre_ls = []
        self.sig_m_pre_ls = []
        self.sig_p_post_ls = []
        self.sig_m_post_ls = []
        decay_ops = []
        dephasing_ops = []
        for i in range(len(self.sig_plus_ls)):
            sig_m = self.sig_minus_ls[i]
            sig_p = self.sig_plus_ls[i]
            num = sig_p * sig_m
            
            
            sig_p_pre = qt.spre(sig_p)
            single_shape = tf.cast(tf.constant(sig_p_pre), dtype=self.dtype).shape

            self.sig_p_pre_ls.append(tf.broadcast_to(tf.cast(tf.constant(sig_p_pre), dtype=self.dtype)[tf.newaxis,:], shape=(self.batchsize,*single_shape)))
            sig_m_pre = qt.spre(sig_m).full()
            self.sig_m_pre_ls.append(tf.broadcast_to(tf.cast(tf.constant(sig_m_pre), dtype=self.dtype)[tf.newaxis,:], shape=(self.batchsize,*single_shape)))
            sig_p_post = qt.spost(sig_p).full()
            self.sig_p_post_ls.append(tf.broadcast_to(tf.cast(tf.constant(sig_p_post), dtype=self.dtype)[tf.newaxis,:], shape=(self.batchsize,*single_shape)))
            sig_m_post = qt.spost(sig_m).full()
            self.sig_m_post_ls.append(tf.broadcast_to(tf.cast(tf.constant(sig_m_post), dtype=self.dtype)[tf.newaxis,:], shape=(self.batchsize,*single_shape)))
    


    def build_hamiltonian(self, H_vars):
        free_vars = tf.cast((((H_vars[:,:self.N_qubits]+1) * self.omega_max))[:,:,tf.newaxis,tf.newaxis], dtype=self.dtype)
        amp_vars = tf.cast((H_vars[:,self.N_qubits:2*self.N_qubits] * self.omega_max)[:,:,tf.newaxis,tf.newaxis], dtype=self.dtype)
        coup_vars = tf.cast((H_vars[:,2 * self.N_qubits:] * self.g_max)[:,:,tf.newaxis,tf.newaxis], dtype=self.dtype)
        H_pre_ls = []
        H_post_ls = []
        # Free terms
        for j in range(self.N_qubits):
            free_op_pre = tf.matmul(self.sig_p_pre_ls[j],self.sig_m_pre_ls[j])
            free_op_post = tf.matmul(self.sig_m_post_ls[j],self.sig_p_post_ls[j])
            H_pre_ls.append((free_vars[:,j]) * free_op_pre)
            H_post_ls.append((free_vars[:,j]) * free_op_post)
        # Drive terms 
        for l in range(self.N_qubits):
            drive_op_pre = (self.sig_p_pre_ls[l] + self.sig_m_pre_ls[l])
            drive_op_post = (self.sig_m_post_ls[l] + self.sig_p_post_ls[l])
            H_pre_ls.append((amp_vars[:,l]) * drive_op_pre)
            H_post_ls.append((amp_vars[:,l]) * drive_op_post)
        
        # Coupling terms
        for i, idxs in enumerate(self.coup_indices):
            pre_coup_op = (tf.matmul(self.sig_p_pre_ls[idxs[0]],self.sig_m_pre_ls[idxs[1]])
                           + tf.matmul(self.sig_p_pre_ls[idxs[1]], self.sig_m_pre_ls[idxs[0]]))
            # Order of operators in post swap because (AB)T = BT AT
            post_coup_op = (tf.matmul(self.sig_m_post_ls[idxs[0]],self.sig_p_post_ls[idxs[1]])
                           + tf.matmul(self.sig_m_post_ls[idxs[1]], self.sig_p_post_ls[idxs[0]]))
            H_pre_ls.append((coup_vars[:,i]) * pre_coup_op)
            H_post_ls.append((coup_vars[:,i]) * post_coup_op)
        
        H_pre = sum(H_pre_ls)
        H_post = sum(H_post_ls)
        H = -1j * tf.cast((H_pre - H_post), dtype=self.dtype)
        return H

    """
    build dissipator function
    """
    def build_A(self, Vars):
        if self.data=="real":
            size = (self.local_basis_size - 1)
            # separate diagonal elements
            mask = tf.linalg.diag(tf.zeros(size, dtype=tf.float64), padding_value=1)
            # build a lower triangular form with real diagonals
            real_lower_tri = tfp.math.fill_triangular(Vars)
            diag = tf.linalg.diag(tf.abs(tf.linalg.diag_part(real_lower_tri)))
            # get the upper triangular half with zeros on the diagonal, to act as the complex part
            # im_lower_tri = (tf.linalg.LinearOperatorLowerTriangular(tf.transpose(Vars, perm=[0,1,3,2])).to_dense() - diag_matrix)
            # build L and L_dag
            L = real_lower_tri*mask + diag
            L_dag = tf.linalg.adjoint(L)
            # get A
            A = tf.matmul(L,L_dag)
            A/=tf.reduce_max(tf.math.real(A))
        else:
            # separate diagonal elements
            diag_elems = tf.linalg.diag_part(Vars)
            # make a matrix with these on the diag
            diag_matrix = tf.linalg.diag(diag_elems) #+ tf.linalg.diag([0.001] * Var.shape[0])
            # build a lower triangular form with real diagonals
            real_lower_tri = (tf.linalg.LinearOperatorLowerTriangular(Vars).to_dense() - diag_matrix + tf.abs(diag_matrix))
            # get the upper triangular half with zeros on the diagonal, to act as the complex part
            im_lower_tri = (tf.linalg.LinearOperatorLowerTriangular(tf.transpose(Vars, perm=[0,1,3,2])).to_dense() - diag_matrix)  
            # build L and L_dag
            L = tf.complex(real=real_lower_tri, imag=im_lower_tri)
            L_dag = tf.linalg.adjoint(L)
            # get A
            A = tf.cast(tf.matmul(L,L_dag), dtype=self.dtype)
            # A/=tf.cast(tf.reduce_max(tf.abs(A)), 
            #                      dtype=self.dtype)
        return A

    def hermicity_check(self, A):
        x = tf.reduce_sum(tf.abs(A - tf.linalg.adjoint(A)))
        if x<1e-3:
            return True
        else:
            return False

    def positivity_check(self, A):
        eigs = tf.linalg.eigvals(A)
        neg = tf.reduce_sum((tf.abs(tf.math.real(eigs)) - tf.math.real(eigs))/2)
        imaginariness = tf.reduce_sum(tf.abs(tf.math.imag(eigs)))
        if neg<1e-4:
            return True
        else:
            return False



    def build_dissipator(self, A_ls, decoherence=False):
        """
        A function for building the first standard form of the lindblad Dissipator, from a/a list of 
        parameter matrices A_ls corresponding to the values for each basis element in the Fock Liouville
        basis lists from above
        """
        broacasted_A = tf.cast(tf.broadcast_to(A_ls[:,:,:,:,tf.newaxis, tf.newaxis], shape=(*A_ls.shape,*self.sample_shape)), dtype=self.dtype)
        local_contributions = (self.g_max * broacasted_A * tf.cast(self.FL_basis_full, dtype=self.dtype))
        # summing over axis 1,2 sums all the contibutions within one dissipator
        local_dissipative_terms = tf.reduce_sum(local_contributions, axis=[2,3])
        # summing over axis zero adds the terms for each dissipator
        full_dissipative_terms = tf.reduce_sum(local_dissipative_terms, axis=[1])
        return full_dissipative_terms
