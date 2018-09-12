import numpy as np
import copy
import math
import os
# import scipy.optimize
# import matplotlib as mpl
# import scipy.signal
# import matplotlib.pyplot as plt

# mpl.rcParams["savefig.directory"] = "."
np.set_printoptions(precision=7, edgeitems=30, linewidth=100000)

def generate_spin_matrices(spin):
    """ Generateenerates the x, y, z spin matrices for an arbitrary spin"""

    if spin == 0:
        return np.array((0,)), np.array((0,)), np.array((0,))

    # Dimension of spin matrices of a given spin
    dim = int(2 * spin + 1)
    # Generates a descending list from spin to -spin in steps of -1
    desc = np.arange(spin, -spin-1, -1)

    # Create a template for the spin matrices
    sx = np.zeros((dim, dim), dtype=complex)
    sy = np.zeros((dim, dim), dtype=complex)
    sz = np.zeros((dim, dim), dtype=complex)

    # Generate the spin matrices using the formula
    # http://easyspin.org/documentation/spinoperators.html
    for row in range(dim):
        sz[row, row] = desc[row]

        if row == 0:
            sx[row, row+1] = 1./2*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sy[row, row+1] = -(1.j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
        elif row == dim-1:
            sx[row, row-1] = 1./2*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
            sy[row, row-1] = (1.j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
        else:
            sx[row, row+1] = 1./2*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sx[row, row-1] = 1./2*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
            sy[row, row+1] = -(1.j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sy[row, row-1] = (1.j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])

    return sx, sy, sz

class Neighbour:
    def __init__(self, disp_vector, spinH, g_n_host):
        self.disp_vector = 10**-10 * np.array(disp_vector)
        self.R = np.linalg.norm(self.disp_vector)
        self.spinH = spinH
        self.multH = int(2*spinH+1)
        self.g_n_host = g_n_host

        self.Hx, self.Hy, self.Hz = generate_spin_matrices(self.spinH)
        self.H = np.array([self.Hx, self.Hy, self.Hz])
        self.Hv3 = self.H.reshape((3, self.multH**2))

class SpinSystem:

    mu_b = 9.2740100*10**-24 # Bohr Magneton (J/T)
    h = 6.62607004*10**-34   # Planck's constant (J s)
    beta_el = mu_b / (h * 10**9) # Bohr Magneton in GHz/T

    mu_n = 5.0507837*10**-27 # Nuclear Magneton (J/T)
    beta_n = mu_n / (h * 10**9) # Nuclear Magneton in GHz/T

    def unit_vector(self, theta, phi):
        return np.array([ np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),    
                          np.cos(theta)])

    def __init__(self, spinS, spinI, g_gs, g_es, A_gs, A_es, g_n_rei):
        # spinS = Electron spin, spinI = Nuclear spin, spinH = Nuclear spin of host atom
        self.spinS = spinS
        self.spinI = spinI

        # gs = ground state, es = excited state
        # g = electronic zeeman coefficients (3x3 matrix)
        # g_n = REI Nuclear g coefficient (isotropic)
        # g_n_host = Nuclear magnetic moment coefficient of host atom
        self.g_gs = g_gs
        self.g_es = g_es
        self.g_n_rei = g_n_rei

        # A = Hyperfine coefficients (3x3 matrix)
        self.A_gs = A_gs
        self.A_es = A_es

        # Multiplicities of the spin dimensions
        self.multS = int(2*self.spinS+1)
        self.multI = int(2*self.spinI+1)

        # Populate the spin matrices 
        self.sx, self.sy, self.sz = generate_spin_matrices(self.spinS)
        # Concatenate them together. Dimensions: 3 x multS x multS
        self.S = np.array([self.sx, self.sy, self.sz])
        # Convert to a 3D vector with dimensions 3 x multS^2
        self.Sv3 = self.S.reshape((3, self.multS**2))

        # Do the same for the nuclear spin
        self.Ix, self.Iy, self.Iz = generate_spin_matrices(self.spinI)
        self.I = np.array([self.Ix, self.Iy, self.Iz])
        self.Iv3 = self.I.reshape((3, self.multI**2))

        self.neighbours_input = [
        # Generated from VESTA for YVO structure from Y to other Y's or V's
        # ((-3.55915, 0.0, 1.57232), 1./2, -0.2748308), # testing
        # 
        ((0.0, 0.0, -3.14465), 7./2, 1.4711), # V, 3.14465 
        ((0.0, 0.0, 3.14465), 7./2, 1.4711),  # V, 3.14465

        ((-3.55915, 0.0, 1.57232), 1./2, -0.2748308), # Y, 3.89098
        ((3.55915, 0.0, 1.57232), 1./2, -0.2748308), # Y, 3.89098
        # ((0.0, -3.55915, -1.57232), 1./2, -0.2748308), # Y, 3.89098
        # ((0.0, 3.55915, -1.57232), 1./2, -0.2748308), # Y, 3.89098

        # (0.0, -3.55915, 1.57232), # V, 3.89098
        # (0.0, 3.55915, 1.57232), # V, 3.89098
        # (-3.55915, 0.0, -1.57232), # V, 3.89098
        # (3.55915, 0.0, -1.57232), # V, 3.89098

        # (-3.55915, -3.55915, 0.0), # V, 5.0334
        # (-3.55915, 3.55915, 0.0), # V, 5.0334
        # (3.55915, -3.55915, 0.0), # V, 5.0334
        # (3.55915, 3.55915, 0.0), # V, 5.0334

        # (-3.55915, 0.0, -4.71697), # Y, 5.90909
        # (3.55915, 0.0, -4.71697), # Y, 5.90909
        # (0.0, -3.55915, 4.71697), # Y, 5.90909
        # (0.0, 3.55915, 4.71697), # Y, 5.90909
        # (0.0, -3.55915, -4.71697), # V, 5.90909
        # (0.0, 3.55915, -4.71697), # V, 5.90909
        # (-3.55915, 0.0, 4.71697), # V, 5.90909
        # (3.55915, 0.0, 4.71697), # V, 5.90909
        ]

        # Take the list of raw inputs and make it into a list of Neighbour objects
        self.neighbours = [Neighbour(*neigh) for neigh in self.neighbours_input]

        # Hamiltonian currently consists of only the hyperfine and SHF pieces 
        # since they are independent of the B field.
        self.initialize_HHF()
        self.initialize_SHF()
        self.reset_ham()

        # # Update list of energy levels for each state
        self.calc_energies()

    def set_vector(self, disp_vector):
        self.__init__(self.spinS, self.spinI, self.g_gs, self.g_es, self.A_gs, self.A_es, self.g_n_rei, self.g_n_host, self.spinH, disp_vector)

    def reset_ham(self):
        """ Sets the Hamiltonian to consist of only the field independent hyperfine 
        and superhyperfine parts. """

        self.H_gs = copy.deepcopy(self.HHF_gs + self.H_SHF_gs)
        self.H_es = copy.deepcopy(self.HHF_es + self.H_SHF_es)

    def initialize_HHF(self):
        # Compute the hyperfine Hamiltonian by I @ A @ S for the ground state
        # Reshape S from a column matrix of 3 multS x multS matrices to become 3 rows and multS^2 columns
        # This lets to dot with A which is a 3x3 matrix. We then immediately reshape back.
        # Then we take the dot product with I by kronecker product componentwise and sum.
        self.HHF_gs = (self.A_gs.dot(self.Sv3)).reshape((3, self.multS, self.multS))
        self.HHF_gs = sum(np.kron(self.I[i], self.HHF_gs[i]) for i in range(3))

        # Expand into the host nucleus space
        for neigh in self.neighbours: 
            self.HHF_gs = np.kron(self.HHF_gs, np.identity(neigh.multH))

        # Repeat for the excited state
        self.HHF_es = (self.A_es.dot(self.Sv3)).reshape((3, self.multS, self.multS))
        self.HHF_es = sum(np.kron(self.I[i], self.HHF_es[i]) for i in range(3))
        # Expand into the host nucleus space
        for neigh in self.neighbours: 
            self.HHF_es = np.kron(self.HHF_es, np.identity(neigh.multH))

    def initialize_SHF(self):
        self.H_SHF_gs = 0
        self.H_SHF_es = 0

        # mu = mu_b * g * S
        # Use v3 so that we can take the product with the 3x3 g matrix
        # Then reshape so that we get back our column 3 x (multS x multS) 
        mu_REI_gs = -(self.mu_b * self.g_gs.dot(self.Sv3)).reshape((3, self.multS, self.multS)) 
        mu_REI_es = -(self.mu_b * self.g_es.dot(self.Sv3)).reshape((3, self.multS, self.multS))
        # Reshape both back to v3 since we need another dot product, this time with R
        mu_REI_gs_v3 = mu_REI_gs.reshape((3, self.multS**2))
        mu_REI_es_v3 = mu_REI_es.reshape((3, self.multS**2))

        # First term is (mu_REI)dot(mu_host)/R^3
        first_term_gs = 0
        first_term_es = 0
        # Second term is 3(mu_REI dot R)(mu_host dot R)/R^5
        second_term_gs = 0
        second_term_es = 0

        for neigh_idx, neigh in enumerate(self.neighbours):

            mu_host = self.mu_n * neigh.g_n_host * neigh.H
            # Reshape both back to v3 since we need another dot product later with R
            mu_host_v3 = mu_host.reshape((3, neigh.multH**2))

            # Do the dot product of mu_REI dot mu_host but with intervening identities
            for i in range(3):
                gs_term = np.kron(mu_REI_gs[i], np.identity(self.multI))
                es_term = np.kron(mu_REI_es[i], np.identity(self.multI))

                for inner_neigh_idx, inner_neigh in enumerate(self.neighbours):
                    if inner_neigh_idx == neigh_idx:
                        gs_term = np.kron(gs_term, mu_host[i])
                        es_term = np.kron(es_term, mu_host[i])
                    else:
                        gs_term = np.kron(gs_term, np.identity(inner_neigh.multH))
                        es_term = np.kron(es_term, np.identity(inner_neigh.multH))

                first_term_gs += gs_term / neigh.R**3
                first_term_es += es_term / neigh.R**3

            # Take the dot product then reshape back
            dot_REI_gs = neigh.disp_vector.dot(mu_REI_gs_v3).reshape((self.multS, self.multS))
            dot_REI_es = neigh.disp_vector.dot(mu_REI_es_v3).reshape((self.multS, self.multS))
            dot_host = neigh.disp_vector.dot(mu_host_v3).reshape((neigh.multH, neigh.multH))
            # Expand into the REI nuclear space first
            dot_REI_gs = np.kron(dot_REI_gs, np.identity(self.multI))
            dot_REI_es = np.kron(dot_REI_es, np.identity(self.multI))

            # Second term is 3(mu_REI dot R)(mu_host dot R)/R^5
            for inner_neigh_idx, inner_neigh in enumerate(self.neighbours):
                if inner_neigh_idx == neigh_idx:
                    dot_REI_gs = np.kron(dot_REI_gs, dot_host)
                    dot_REI_es = np.kron(dot_REI_es, dot_host)
                else:
                    dot_REI_gs = np.kron(dot_REI_gs, np.identity(inner_neigh.multH))
                    dot_REI_es = np.kron(dot_REI_es, np.identity(inner_neigh.multH))

            second_term_gs += (3/neigh.R**5) * dot_REI_gs
            second_term_es += (3/neigh.R**5) * dot_REI_es

        # Divide by the constant factor to convert from J to GHz
        # First 10**-7 comes from mu0/4pi
        self.H_SHF_gs = 10**-7 * (first_term_gs - second_term_gs) / (self.h * 10**9)
        self.H_SHF_es = 10**-7 * (first_term_es - second_term_es) / (self.h * 10**9)

    def update_zeeman_ham(self, B, B_theta, B_phi):
        """ Given a (new) value of the magnetic field, compute the new Zeeman
        interaction term in the hamiltonian and then update the hamiltonian. """

        # Convert B into cartesian
        B_vec = B * np.array([  np.sin(B_theta) * np.cos(B_phi),
                                np.sin(B_theta) * np.sin(B_phi),    
                                np.cos(B_theta)])

        # Compute the electronic Zeeman Hamiltonian using beta * B @ g @ S
        HZ_el_gs = (self.beta_el * B_vec.dot(self.g_gs).dot(self.Sv3)).reshape((self.multS, self.multS))
        HZ_el_es = (self.beta_el * B_vec.dot(self.g_es).dot(self.Sv3)).reshape((self.multS, self.multS))
        # Expand into the nuclear spin space
        HZ_el_gs = np.kron(HZ_el_gs, np.identity(self.multI))
        HZ_el_es = np.kron(HZ_el_es, np.identity(self.multI))
        # Then expand into the superhyperfine nuclear spin space
        for neigh in self.neighbours: 
            HZ_el_gs = np.kron(HZ_el_gs, np.identity(neigh.multH))
            HZ_el_es = np.kron(HZ_el_es, np.identity(neigh.multH))

        # Do the same for the REI nuclear Zeeman Hamiltonian but there is no 
        # distinction between GS and ES. We assume the nuclear g factor is an 
        # isotropic scalar.
        HZ_n_rei = (self.beta_n * self.g_n_rei * B_vec.dot(self.Iv3)).reshape((self.multI, self.multI))
        # Expand into the electronic spin space
        HZ_n_rei = np.kron(np.identity(self.multS), HZ_n_rei)
        # Expand into the superhyperfine nuclear spin space
        for neigh in self.neighbours: 
            HZ_n_rei = np.kron(HZ_n_rei, np.identity(neigh.multH))

        # Do the same for the nuclear Zeeman Hamiltonian  for the host nuclear spin
        HZ_n_host = 0
        for neigh_idx, neigh in enumerate(self.neighbours):
            # Start with a blank template for the S and I spaces
            HZ_n_host_accum = np.kron(np.identity(self.multS), np.identity(self.multI))
            HZ_n_host_term = (self.beta_n * neigh.g_n_host * B_vec.dot(neigh.Hv3)).reshape((neigh.multH, neigh.multH))

            # Iterate through the neighbours and expand into their spaces
            # When it's its own index we cross the actual Zeeman interaction instead
            for inner_neigh_idx, inner_neigh in enumerate(self.neighbours):
                if inner_neigh_idx == neigh_idx:
                    HZ_n_host_accum = np.kron(HZ_n_host_accum, HZ_n_host_term)
                else:
                    HZ_n_host_accum = np.kron(HZ_n_host_accum, np.identity(inner_neigh.multH))

            HZ_n_host += HZ_n_host_accum

        # Reset the Hamiltonian to be only the field-indep HF part and SHF part
        self.reset_ham()
        # Add in the just-computed Zeeman terms
        self.H_gs += (HZ_el_gs - HZ_n_rei - HZ_n_host)
        self.H_es += (HZ_el_es - HZ_n_rei - HZ_n_host)

    def update_B(self, B, B_theta, B_phi):
        """ Updates the Zeeman Hamiltonian contribution from a newly
        specified B field with magnitude B and directions theta, phi. The Zeeman
        Hamiltonian is then added to the state's Hamiltonian and the energies 
        are updated. """

        # Include the Zeeman part (REI electron + REI nucleus + host nucleus)
        self.update_zeeman_ham(B, B_theta, B_phi)
        # Update energies
        # print("Computing eigenvalues...")
        self.calc_energies()
        # print("Completed.")

    def calc_energies(self):
        """ Compute energies for the current Hamiltonian and updates state."""

        # Get the ground state eigvals and eigvecs
        eigval_gs, eigvec_gs = np.linalg.eig(self.H_gs)
        # Sort the columns of the eigvecs in increasing size of the eigvals
        self.E_gs = sorted(eigval_gs)
        self.eigvec_gs = eigvec_gs[:, np.argsort(eigval_gs)]

        # Do the same for the excited state
        eigval_es, eigvec_es = np.linalg.eig(self.H_es)
        # Sort the columns of the eigvecs in increasing size of the eigvals
        self.E_es = sorted(eigval_es)
        self.eigvec_es = eigvec_es[:, np.argsort(eigval_es)]

    def energies(self, state, B=None, B_theta=None, B_phi=None):
        """ Returns energies of a system with the hyperfine interaction and 
        electronic Zeeman interaction. Optional to provide a B field to update 
        before returning. """
        
        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        if state == 'gs':
            return self.E_gs
        elif state == 'es':
            return self.E_es
        else:
            raise Exception

    def compute_holeburning_spectrum(self, B, B_theta, B_phi, axis, gs_blobs, es_blobs, atoms):
        # Update the B field 
        self.update_B(B, B_theta, B_phi)

        gs_blob_lower, gs_blob_upper = gs_blobs
        es_blob_lower, es_blob_upper = es_blobs
        gs_blob = list(gs_blob_lower) + list(gs_blob_upper)
        es_blob = list(es_blob_lower) + list(es_blob_upper)
        transitions_lower = [(i, j) for i in gs_blob_lower for j in es_blob_lower]
        transitions_upper = [(i, j) for i in gs_blob_upper for j in es_blob_upper]
        burn_transitions = transitions_lower + transitions_upper

        axis_dict = {0: "x", 1: "y", 2: "z"}
        S_op = {0: self.sx, 1 : self.sy, 2 : self.sz}

        # Set the axis for the E field polarisation to use the appropriate S matrix
        # for the transition probabilities
        bigS_op = 2*np.kron(S_op[axis], np.identity(self.multI))
        # Expand into neighbour space
        for neigh in self.neighbours:
            bigS_op = np.kron(bigS_op, np.identity(neigh.multH))
        
        spectrum_data = []
        usable_transitions = []

        # Memoisation of the intensities
        print("Computing intensities...")
        cross_intensity_matrix = np.zeros((len(gs_blob), len(es_blob)))
        ground_intensity_matrix = np.zeros((len(gs_blob), len(gs_blob)))

        # for gs_idx, es_idx in burn_transitions:
        for es_idx in gs_blob:
            es_vec = self.eigvec_es[:, es_idx]
            for gs_idx in es_blob:
                gs_vec = self.eigvec_gs[:, gs_idx]
                intensity = abs(es_vec.conjugate().dot(bigS_op).dot(gs_vec.T))**2
                if intensity > 10**-8:
                    cross_intensity_matrix[es_idx][gs_idx] = intensity
                    usable_transitions.append((gs_idx, es_idx))

        print("half done...")
        for gs_upper_idx in gs_blob:
            gs_upper_vec = self.eigvec_gs[:, gs_upper_idx]
            for gs_lower_idx in gs_blob:
                gs_lower_vec = self.eigvec_gs[:, gs_lower_idx]
                ground_intensity_matrix[gs_upper_idx][gs_lower_idx] = abs(gs_upper_vec.conjugate().dot(bigS_op).dot(gs_lower_vec.T))**2
                # ground_intensity_matrix[gs_lower_idx][gs_upper_idx] = abs(gs_upper_vec.conjugate() @ bigS_op @ gs_lower_vec.T)**2
        print("Done!")

        cross_intensity_matrix /= np.sum(cross_intensity_matrix, axis=1, keepdims=True)
        ground_intensity_matrix /= np.sum(ground_intensity_matrix,axis=1, keepdims=True)
        
        # lower and upper burn set refer to the choice of branch for the Zeeman
        # and hyperfine energy levels that we are concerned about. We don't
        # care about any population/transitions from those other manifolds.
        count = 0
        for gs_burn_idx, es_burn_idx in burn_transitions:
            print("Processing inhomogeneous pair {}/{}: ({}, {})".format(count, len(burn_transitions), gs_burn_idx, es_burn_idx))
            count += 1

            # Energy of the holeburning pumping transition
            central_energy = self.E_es[es_burn_idx] - self.E_gs[gs_burn_idx]
            # if not 60.025 < central_energy < 60.155: continue
            
            # States that have high chance of transitioning to the burnt state
            # They will get burnt away too because the burnt state becomes empty
            gs_reswith_burnt = [gs_idx for gs_idx in gs_blob if ground_intensity_matrix[gs_idx][gs_burn_idx] > 10**-5]

            # Iterate through all levels in GS and find prob of transitioning 
            # from es_burn_idx. We exclude the gs_reswith_burnt states since they're
            # being burned away.
            gs_populations = [cross_intensity_matrix[es_burn_idx][gs_idx] if gs_idx not in gs_reswith_burnt \
                                                                        else 0 for gs_idx in gs_blob]

            # Normalise the population ratios to be equal to the number of 
            # burnt states.
            # if np.sum(gs_populations) != 0: 
            #     gs_populations /= np.sum(gs_populations)
            #     # gs_populations *= len(gs_reswith_burnt)
            # else:
            #     print("Error in transition {}->{}".format(gs_burn_idx, es_burn_idx))         

            # First create the holes for transitions starting from gs_reswith_burnt 
            for gs_idx in gs_reswith_burnt:
                for es_idx in es_blob:
                    energy = (self.E_es[es_idx] - self.E_gs[gs_idx] - central_energy).real
                    intensity = cross_intensity_matrix[es_idx][gs_idx]
                    # Normalise with respect to the number of inhomog pairs we are looking at
                    intensity /= len(burn_transitions)

                    if intensity > 10**-8:
                        spectrum_data.append((gs_idx, es_idx, energy, intensity))

            # Next create the antiholes for transitions starting from the gs_idx
            for gs_idx, es_idx in usable_transitions:
            # for gs_idx in gs_blob:
                # There are no antiholes from the burnt level (would actually have  
                # been taken care off by the gs_populations list anyway)
                if gs_idx in gs_reswith_burnt or gs_populations[gs_idx] == 0: 
                    continue
                # for es_idx in es_blob: 
                energy = (self.E_es[es_idx] - self.E_gs[gs_idx] - central_energy).real
                # Scale by both the absorption strength and the population size
                intensity = cross_intensity_matrix[es_idx][gs_idx] * gs_populations[gs_idx]
                # Normalise with respect to the number of inhomog pairs we are looking at
                intensity /= len(burn_transitions)

                if intensity > 10**-8:
                    spectrum_data.append((gs_idx, es_idx, energy, -intensity))

        # spectrum_data.sort()
        np.savetxt("SpectrumData_{}_{}_{}_{}_pol_{}_neighs_{}_fall_nonorm_pypy.txt".format(B, B_theta, B_phi, axis_dict[axis], len(self.neighbours), atoms), 
            spectrum_data, 
            header="gs_idx, es_idx, energy, intensity",
            fmt=("%d", "%d", "%.4e", "%.4e"))


        return spectrum_data

################################################################################

yb171_A_gs = np.array([[0.6745, 0,  0], #0.675, -4.82, from Ranon 1968, Kindem 2018
                       [0, 0.6745,   0],
                       [0, 0,  -4.8205]])


yb171_A_es = np.array([[3.39,   0,  0], # 3.37, 4.86, from Kindem 2018
                        [0, 3.39,  0],
                        [0,   0,  4.864]])

yb171_g_gs = np.array([[0.85, 0,  0], # From Ranon 1968
                       [0,  0.85,  0],
                       [0,  0, -6.08]])

yb171_g_es = np.array([[1.6, 0,  0], # 1.7, 2.51, from Kindem 2018
                       [0, 1.6,  0],
                       [0,  0, 2.51]])

electron_zeeman = 1
hyperfine = 1
nuclear_zeeman_rei = 1

A = SpinSystem(
    1./2,                                 # Yb-171                   # REI Electronic Spin
    nuclear_zeeman_rei * 1./2,            # Yb-171                   # REI Nuclear Spin
    electron_zeeman * yb171_g_gs,                                   # REI Electronic GS g matrix
    electron_zeeman * yb171_g_es,                                   # REI ELectronic ES g matrix
    hyperfine * yb171_A_gs,                                         # GS Hyperfine A matrix
    hyperfine * yb171_A_es,                                         # ES Hyperfine A matrix
    nuclear_zeeman_rei * 0.98734,       # Yb-171                  # REI Nuclear g coeff
    )
print("Done setting up!")
################################################################################
#
mult = 256
A.compute_holeburning_spectrum(0.721, 0.33929, 0, 0, (range(mult), range(mult,2*mult)), (range(mult), range(mult,2*mult)), atoms="VVYY")