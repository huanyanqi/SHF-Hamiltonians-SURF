import os
import copy
import bisect
import numpy as np
import matplotlib as mpl
import scipy.signal
import matplotlib.pyplot as plt

mpl.rcParams["savefig.directory"] = "."
np.set_printoptions(precision=7, edgeitems=30, linewidth=100000)
mpl.rcParams.update({'font.size': 24})

def generate_spin_matrices(spin):
    """ Generates the x, y, z spin matrices for an arbitrary spin"""
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
            sx[row, row+1] = 1/2*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sy[row, row+1] = -(1j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
        elif row == dim-1:
            sx[row, row-1] = 1/2*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
            sy[row, row-1] = (1j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
        else:
            sx[row, row+1] = 1/2*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sx[row, row-1] = 1/2*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])
            sy[row, row+1] = -(1j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row+1])
            sy[row, row-1] = (1j/2)*np.sqrt(spin*(spin+1)-desc[row]*desc[row-1])

    return sx, sy, sz

class Neighbour:
    def __init__(self, disp_vector, spinH, g_n_host, element):
        self.disp_vector = 10**-10 * np.array(disp_vector)
        self.R = np.linalg.norm(self.disp_vector)
        self.spinH = spinH
        self.multH = int(2*spinH+1)
        self.g_n_host = g_n_host
        self.element = element

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

    def set_vector(self, disp_vector):
        self.__init__(self.spinS, self.spinI, self.g_gs, self.g_es, self.A_gs, self.A_es, self.g_n_rei, self.g_n_host, self.spinH, disp_vector)

    def __init__(self, spinS, spinI, g_gs, g_es, A_gs, A_es, g_n_rei, neighbours_list):
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

        self.neighbours_input = neighbours_list 
        # Take the list of raw inputs and make it into a list of Neighbour objects
        self.neighbours = [Neighbour(*neigh) for neigh in self.neighbours_input]
        self.neighbours_string = "".join([neigh.element for neigh in self.neighbours])

        # Hamiltonian currently consists of only the hyperfine and SHF pieces 
        # since they are independent of the B field.
        self.initialize_HHF()
        self.initialize_SHF()
        self.reset_ham()

        # # Update list of energy levels for each state
        self.calc_energies()

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
        self.HHF_gs = (self.A_gs @ self.Sv3).reshape((3, self.multS, self.multS))
        self.HHF_gs = sum(np.kron(self.I[i], self.HHF_gs[i]) for i in range(3))

        # Expand into the host nucleus space
        for neigh in self.neighbours: 
            self.HHF_gs = np.kron(self.HHF_gs, np.identity(neigh.multH))

        # Repeat for the excited state
        self.HHF_es = (self.A_es @ self.Sv3).reshape((3, self.multS, self.multS))
        self.HHF_es = sum(np.kron(self.I[i], self.HHF_es[i]) for i in range(3))
        # Expand into the host nucleus space
        for neigh in self.neighbours: 
            self.HHF_es = np.kron(self.HHF_es, np.identity(neigh.multH))

    def initialize_SHF(self):
        # mu = mu_b * g * S
        # Use v3 so that we can take the product with the 3x3 g matrix
        # Then reshape so that we get back our column 3 x (multS x multS) 
        mu_REI_gs = -(self.mu_b * self.g_gs @ self.Sv3).reshape((3, self.multS, self.multS)) 
        mu_REI_es = -(self.mu_b * self.g_es @ self.Sv3).reshape((3, self.multS, self.multS))
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
            # Reshape back to v3 since we need another dot product later with R
            mu_host_v3 = mu_host.reshape((3, neigh.multH**2))

            # Do the dot product of mu_REI dot mu_host but with intervening identities
            # for the REI nuclear space + spectator host nuclei
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
        HZ_el_gs = (self.beta_el * B_vec @ self.g_gs @ self.Sv3).reshape((self.multS, self.multS))
        HZ_el_es = (self.beta_el * B_vec @ self.g_es @ self.Sv3).reshape((self.multS, self.multS))
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
        HZ_n_rei = (self.beta_n * self.g_n_rei * B_vec @ self.Iv3).reshape((self.multI, self.multI))
        # Expand into the electronic spin space
        HZ_n_rei = np.kron(np.identity(self.multS), HZ_n_rei)
        # Expand into the superhyperfine nuclear spin space
        for neigh in self.neighbours: 
            HZ_n_rei = np.kron(HZ_n_rei, np.identity(neigh.multH))

        # Do the same for the nuclear Zeeman Hamiltonian for the host nuclear spin
        HZ_n_host = 0
        for neigh_idx, neigh in enumerate(self.neighbours):
            # Start with a blank template for the S and I spaces
            HZ_n_host_accum = np.kron(np.identity(self.multS), np.identity(self.multI))
            HZ_n_host_term = (self.beta_n * neigh.g_n_host * B_vec @ neigh.Hv3).reshape((neigh.multH, neigh.multH))

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
        # Minus for the nuclear terms since U = -mu.B but no minus for electron
        # since the mu we calculated above is actually negative of what it
        # really is.
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
        self.calc_energies()

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

    def spin_transitions(self, B=None, B_theta=None, B_phi=None):
        """ Computes the transitions wit1hin each energy level. Optional to 
        provide a B field to update before computing. Currently it 
        cannot distinguish between the splitting due to hyperfine and those
        due to Zeeman effect. TODO (?) """

        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        gs_transitions = []
        es_transitions = []

        for i in range(len(self.E_es)):
            for j in range(i+1, len(self.E_gs)):
                gs_transitions.append((self.E_gs[j]-self.E_gs[i], (1,1,1))) # 1 is a placeholder for the transition strength
        for i in range(len(self.E_es)):
            for j in range(i+1, len(self.E_es)):
                es_transitions.append((self.E_es[j]-self.E_es[i], (1,1,1)))
        return [gs_transitions, es_transitions]

    def optical_transitions(self, B=None, B_theta=None, B_phi=None, compute_strengths=False):
        """ Computes the transitions between the energy levels. Optional to
        provide a B field to update before computing. """

        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        transitions = []

        for idx_g, e_g in enumerate(self.E_gs):
            for idx_e, e_e in enumerate(self.E_es):
                if compute_strengths:
                    transitions.append((e_e-e_g, self.transition_strength(self.eigvec_gs[:, idx_g], self.eigvec_es[:, idx_e]), (idx_g, idx_e)))
                else:
                    transitions.append((e_e-e_g, None, (idx_g, idx_e)))
        return transitions

    def superhyperfine_levels(self, state, B=None, B_theta=None, B_phi=None):
        """ Looks at the lower electronic zeeman branch of the ground state to 
        look at the superhyperfine energy splittings. Returns the energy of the 
        doublet's upper and lower levels relative to their average as well as 
        the energy gap. """

        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        # Pick the lowest and 2nd lowest energy levels and assume that these
        # represent the energy splitting of the lowest energy state due to 
        # the superhyperfine interaction. 
        low = self.energies(state, B, B_theta, B_phi)[0]
        up = self.energies(state, B, B_theta, B_phi)[1]
        avg = (low + up) / 2

        # Convert from GHz to KHz since this is really small
        lower = (low - avg)  * 10**6
        upper = (up - avg) * 10**6
        diff = (up - low) * 10**6

        # Look at the highest 2 states instead
        low2 = self.energies(state, B, B_theta, B_phi)[-2]
        up2 = self.energies(state, B, B_theta, B_phi)[-1]
        avg2 = (low2 + up2) / 2

        # Convert from GHz to KHz since this is really small
        lower2 = (low2 - avg2)  * 10**6
        upper2 = (up2 - avg2) * 10**6
        diff2 = (up2 - low2) * 10**6

        return lower, upper, diff, lower2, upper2, diff2

    def branching_contrast(self, B=None, B_theta=None, B_phi=None, stated_levels=None):
        """ Computes the branching contrast of the system (given its current 
        displacement vector of the host atom) using the formula given in Car. """
        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        bigSx = 2*np.kron(np.kron(self.sx, np.identity(self.multI)), np.identity(self.multH))
        bigSy = 2*np.kron(np.kron(self.sy, np.identity(self.multI)), np.identity(self.multH))
        bigSz = 2*np.kron(np.kron(self.sz, np.identity(self.multI)), np.identity(self.multH))

        if stated_levels is None:
            gs_lower = self.eigvec_gs[:, 0]
            gs_upper = self.eigvec_gs[:, 1]
            es_lower = self.eigvec_es[:, 0]
        else:
            i, j, k = stated_levels
            gs_lower = self.eigvec_gs[:, i]
            gs_upper = self.eigvec_gs[:, j]
            es_lower = self.eigvec_es[:, k]

        # Overlap while enclosing the Sx, Sy, Sz matrices
        r_23_S_x = abs(gs_upper.conjugate() @ bigSx @ es_lower.T)**2
        r_13_S_x = abs(gs_lower.conjugate() @ bigSx @ es_lower.T)**2

        RS_x = r_23_S_x / r_13_S_x
        rho_S_x = 4 * RS_x / (1 + RS_x)**2

        return r_23_S_x, r_13_S_x, RS_x, rho_S_x, gs_lower, gs_upper, es_lower

    def transition_strength(self, initial, final):
        """ Computes the transition strength for two eigenvectors initial and
        final for the 3 possible polarisation axes. """

        # Transition operator for the 3 axes (expanded into REI nuclear)
        bigSx = 2*np.kron(self.sx, np.identity(self.multI))
        bigSy = 2*np.kron(self.sy, np.identity(self.multI))
        bigSz = 2*np.kron(self.sz, np.identity(self.multI))
        # Expand into host spaces
        for neigh in self.neighbours:
            bigSx = np.kron(bigSx, np.identity(neigh.multH))
            bigSy = np.kron(bigSy, np.identity(neigh.multH))
            bigSz = np.kron(bigSz, np.identity(neigh.multH))
        # Compute the mod square of the inner product
        fx = abs(final.conjugate() @ bigSx @ initial) ** 2
        fy = abs(final.conjugate() @ bigSy @ initial) ** 2
        fz = abs(final.conjugate() @ bigSz @ initial) ** 2
        return fx, fy, fz

############################
    
    def plot_energies_brange(self, B_range, B_theta, B_phi):
        """ Plot all the energy levels of the system as a function of B field 
        strength. """
        plt.figure()
        plt.suptitle("Energy diagram as a function of B-field")
        plt.subplot(211)
        plt.plot(B_range, [self.energies('es', b, B_theta, B_phi) for b in B_range])
        plt.title("Excited state")
        plt.xlabel("B field / T")
        plt.ylabel("Energy / GHz")
        plt.subplot(212)
        plt.plot(B_range, [self.energies('gs', b, B_theta, B_phi) for b in B_range])
        plt.title("Ground state")
        plt.ylabel("Energy / GHz")

    def plot_optical_transitions(self, B_range, B_theta, B_phi):
        """ Plot all the optical transition energies as a function of B field. """
        plt.figure()
        # Take the 0-th component as the optical transitions function returns 
        # (transition_energy, initial_state, final_state) as a tuple.
        plt.plot(B_range, [[transition[0] for transition in self.optical_transitions(b, B_theta, B_phi)] for b in B_range])
        plt.title("Optical Transitions between energy levels")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")

    def plot_transitions_strengths(self, B_range, B_theta, B_phi, transition_type, axis=None):
        """ Plot all the optical transition energies as a function of B field. 
        The colour of the line represents the strength of that transition. """

        # Array in energy space (GHz) at which to compute the transition strength
        if transition_type == "optical":
            E_grid = np.arange(-10, 10, 0.001)
        else:
            E_grid = np.arange(0, 10, 0.001)
        E_grid_size = len(E_grid)
        B_grid_size = len(B_range)

        if axis is None:
            transition_grid_x = np.zeros((E_grid_size, B_grid_size))
            transition_grid_y = np.zeros((E_grid_size, B_grid_size))
            transition_grid_z = np.zeros((E_grid_size, B_grid_size))
        else:
            transition_grid = np.zeros((E_grid_size, B_grid_size))

        for B_idx, B in enumerate(B_range):
            # Update the B field and get the list of possible optical transitions
            # as a list of (transition_energy, (fx, fy, fy)) tuples where
            # fx, fy, fz refer to the transition strength for E field parallel
            # to each of the 3 axes.
            if transition_type == "optical":
                transitions = self.optical_transitions(B, B_theta, B_phi, compute_strengths=True)
            elif transition_type == "spin_gs":
                transitions = self.spin_transitions(B, B_theta, B_phi, compute_strengths=True)[0]
            elif transition_type == "spin_es":
                transitions = self.spin_transitions(B, B_theta, B_phi, compute_strengths=True)[1]
            else:
                raise Exception
            
            if axis is None:
                # Used to represent each column in the grid which we add each peak onto
                grid_col_x = np.zeros(E_grid_size)
                grid_col_y = np.zeros(E_grid_size)
                grid_col_z = np.zeros(E_grid_size)
            else:
                grid_col = np.zeros(E_grid_size)

            # For each possible optical transition at the particular value of 
            # B field, we take the transition energy (0-th comp) and create a 
            # Lorentzian peak around in the E_grid space, then scale it by its
            # amplitude given by transition[1] (x,y,z) comps.
            for transition in transitions:
                # FWHM of the Lorentzian
                fwhm = 0.1

                if axis is None:
                    grid_col_x += (fwhm/2)**2 * transition[1][0] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                    grid_col_y += (fwhm/2)**2 * transition[1][1] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                    grid_col_z += (fwhm/2)**2 * transition[1][2] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                else:
                    grid_col += (fwhm/2)**2 * transition[1][axis] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                
            if axis is None:
                # We reverse the direction of each row since imshow by default plots from top to bottom
                transition_grid_x[:, B_idx] = np.exp(-5 * grid_col_x[::-1])
                transition_grid_y[:, B_idx] = np.exp(-5 * grid_col_y[::-1])
                transition_grid_z[:, B_idx] = np.exp(-5 * grid_col_z[::-1])
            else:
                transition_grid[:, B_idx] = np.exp(-5 * grid_col[::-1])

        # Plot a slice of the intensity at a particular value of B field
        # plt.plot(E_grid, transition_grid[:, 0][::-1])

        if axis is None:
            plt.figure()
            plt.suptitle("{} Transitions between energy levels".format(transition_type.capitalize()))
            plt.subplot(131)
            plt.imshow(transition_grid_x, cmap="bone", extent=(B_range[0], B_range[-1], E_grid[0], E_grid[-1]), aspect='auto')
            plt.title("E field // x")
            plt.xlabel("B field / T")
            plt.ylabel("Transition Energy / GHz")

            plt.subplot(132)
            plt.imshow(transition_grid_y, cmap="bone", extent=(B_range[0], B_range[-1], E_grid[0], E_grid[-1]), aspect='auto')
            plt.title("E field // y")
            plt.xlabel("B field / T")

            plt.subplot(133)
            plt.imshow(transition_grid_z, cmap="bone", extent=(B_range[0], B_range[-1], E_grid[0], E_grid[-1]), aspect='auto')
            plt.title("E field // z")
            plt.xlabel("B field / T")
        else:
            plt.figure()
            plt.suptitle("{} Transitions between energy levels".format(transition_type))
            plt.imshow(transition_grid, cmap="bone", extent=(B_range[0], B_range[-1], E_grid[0], E_grid[-1]), aspect='auto')
            plt.title("E field // " + {0:"x", 1:"y", 2:"z"}[axis])
            plt.xlabel("B field / T")
            plt.ylabel("Transition Energy / GHz")

    def plot_optical_transitions_strengths_fixed_B(self, B, B_theta, B_phi, axis=None):
        """ Plot all the optical transition energies as a function of B field. 
        The colour of the line represents the strength of that transition. """

        # Array in energy space (GHz) at which to compute the transition strength
        E_grid = np.arange(-100, 100, 0.001)
        E_grid_size = len(E_grid)
        # FWHM of the Lorentzian
        fwhm = 0.001

        opt_trans = self.optical_transitions(B, B_theta, B_phi, compute_strengths=True)
        plt.figure()
        
        if axis is None:
            # Used to represent each column in the grid which we add each peak onto
            optical_peaks_x, optical_peaks_y, optical_peaks_z = np.zeros(E_grid_size), np.zeros(E_grid_size), np.zeros(E_grid_size)
        else:
            optical_peaks = np.zeros(E_grid_size)

        # For each possible optical transition at the particular value of 
        # B field, we take the transition energy (0-th comp) and create a 
        # Lorentzian peak around in the E_grid space, then scale it by its
        # amplitude given by transition[1] (x,y,z) comps.
        for idx, transition in enumerate(opt_trans):
            print("{}/{}".format(idx, len(opt_trans)))
            if axis is None:
                optical_peaks_x += (fwhm/2)**2 * transition[1][0] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                optical_peaks_y += (fwhm/2)**2 * transition[1][1] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                optical_peaks_z += (fwhm/2)**2 * transition[1][2] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
            else:
                optical_peaks += (fwhm/2)**2 * transition[1][axis] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )

            # plt.annotate('Transition ({}, {})'.format(*transition[2]), xy=(transition[0], max(transition[1])))
            
        # Plot a slice of the intensity at a particular value of B field
        plt.title("Optical Transition Strengths with (B, B_theta, B_phi) = {0}".format((B, B_theta, B_phi)))
        if axis is None:
            plt.plot(E_grid, optical_peaks_x, '-', label="E // x")
            plt.plot(E_grid, optical_peaks_y, '--', label="E // y")
            plt.plot(E_grid, optical_peaks_z, '-.', label="E // z")

        else:
            plt.plot(E_grid, optical_peaks, label="E // " + {0:"x", 1:"y", 2:"z"}[axis])

        plt.xlabel("Transition Energy Detuning / GHz")
        plt.ylabel("Transition Strength")
        plt.legend()

    def plot_spin_transitions(self, B_range, B_theta, B_phi):
        """ Plot all the spin transition energies as a function of B field. """
        plt.figure()
        plt.suptitle("Spin Transitions within energy level")
        plt.subplot(211)
        plt.plot(B_range, np.array([[transition[0] for transition in self.spin_transitions(b, B_theta, B_phi)[1]] for b in B_range]))
        plt.title("Excited State Transitions")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")
        plt.subplot(212)
        plt.plot(B_range, np.array([[transition[0] for transition in self.spin_transitions(b, B_theta, B_phi)[0]] for b in B_range]))
        plt.title("Ground State Transitions")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")

    def plot_superhyperfine(self, B_range, B_theta, B_phi):
        """ Plot the superhyperfine splitted levels for the lowest energy levels
        (i.e. plot the lowest 2 energy levels after incorporating the SHF
        interaction). In addition, plot the energy gap between these SHF-split
        energy levels.
        """
        plt.figure()
        plt.suptitle("Superhyperfine splittings")

        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax1.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[:2] for b in B_range], c='C0')
        ax1.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[3:5] for b in B_range], c='C2')
        ax1.set_title("Relative Energies of ground superhyperfine doublet")
        ax1.set_xlabel("B field / T")
        ax1.set_ylabel("Relative Energy / kHz")

        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax2.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[:2] for b in B_range], c='C1')
        ax2.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[3:5] for b in B_range], c='C3')
        ax2.set_title("Relative Energies of excited superhyperfine doublet")
        ax2.set_xlabel("B field / T")
        ax2.set_ylabel("Relative Energy / kHz")

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[2] for b in B_range], c='C0', label="Ground state Lower")
        ax3.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[5] for b in B_range], c='C2', label="Ground state Upper")
        ax3.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[2] for b in B_range], c='C1', label="Excited state Lower")
        ax3.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[5] for b in B_range], c='C3', label="Excited state Upper")
        ax3.set_title("Energy gap between the superhyperfine doublet levels")
        ax3.set_xlabel("B field / T")
        ax3.set_ylabel("Transition Energy / kHz")
        ax3.legend()

    def compute_holeburning_spectrum(self, B, B_theta, B_phi, axis, gs_blobs, es_blobs):
        """ Given a fixed B field, axis for the optical E field polarisation, 
        a set of states to burn from (gs_blobs) and to (es_blobs), compute the
        set of transitions (energy and intensity) that result from this spectral
        holeburning. """

        # Update the B field 
        self.update_B(B, B_theta, B_phi)

        # blobs consist of ranges that transition to each other
        # E.g. gs_blobs = (range(2), range(2, 4)), egs_blobs = (range(2), range(2, 4))
        # means that we have transitions 0->0,1 + 1->0,1 + 2->2,3 + 3->2,3
        # We must have the same number of blobs for both GS and ES to match them up
        # Also the elements in each constituent range in the blobs must not overlap
        # since otherwise we could just combine the ranges together (assuming the
        # laser is wide enough).
        assert(len(gs_blobs) == len(es_blobs))

        # Used to store the union of the blobs
        gs_blob = []
        es_blob = []
        burn_transitions = []

        for idx in range(len(gs_blobs)):
            gs_blob_i = gs_blobs[idx]
            es_blob_i = es_blobs[idx]

            gs_blob += list(gs_blob_i)
            es_blob += list(es_blob_i)
            # Only consider transitions within the same type of blob
            burn_transitions += [(i, j) for i in gs_blob_i for j in es_blob_i]

        # Check that the 2 blob consists of only unique elements
        assert len(gs_blob) == len(set(gs_blob))
        assert len(es_blob) == len(set(es_blob))

        # The number of possible burn transitions is equal to the number of inhomogeneous
        # broadening atoms we need to consider.
        num_inhomog_atoms = len(burn_transitions)
        # Intensity threshold for recording a transition in the file (to prevent
        # the filesize from being too large)
        scaled_intensity_threshold = 10**-6 / num_inhomog_atoms

        bigS_op_x = 2*np.kron(self.sx, np.identity(self.multI))
        bigS_op_y = 2*np.kron(self.sy, np.identity(self.multI))
        bigS_op_z = 2*np.kron(self.sz, np.identity(self.multI))
        # Expand into neighbour space
        for neigh in self.neighbours:
            bigS_op_x = np.kron(bigS_op_x, np.identity(neigh.multH))
            bigS_op_y = np.kron(bigS_op_y, np.identity(neigh.multH))
            bigS_op_z = np.kron(bigS_op_z, np.identity(neigh.multH))

        # Set the axis for the E field polarisation to use the appropriate S matrix
        # for the transition probabilities
        axis_dict = {"x": 0, "y": 1, "z": 2}
        bigS_op_dict = {"x": bigS_op_x, "y": bigS_op_y, "z": bigS_op_z} 
        bigS_op_axis = bigS_op_dict[axis]
        
        # Used to store tuples of transitions in the system
        # Format: (gs_idx, es_idx, energy, intensity)
        # Intensity is > 0  for holes, < 0 for antiholes 
        spectrum_data = []
        # Used as a shortcut to see which excited<->ground transitions have decently
        # strong transitions so that we only need to search those when computing
        # antiholes.
        usable_transitions = []

        # Memoisation of the intensities
        print("Computing intensities...")
        # Transitions between excited and ground states
        # Intensity along the E field polarisation specified by axis
        cross_intensity_matrix = np.zeros((len(self.E_es), len(self.E_gs)))
        # Intensity but sums all polarisations
        cross_intensity_matrix_allpol = np.zeros((len(self.E_es), len(self.E_gs)))
        # Energies of the ES->GS transitions
        cross_energies = []
        # Transitions within ground states
        ground_intensity_matrix = np.zeros((len(self.E_gs), len(self.E_gs)))

        # Look at all possible transitions from all excited states to all
        # ground states which is used for the relaxation from excited->ground
        # state after pumping. Also used for the hole absorption strength.
        for gs_idx in gs_blob:
            gs_vec = self.eigvec_gs[:, gs_idx]
            for es_idx in es_blob:
                es_vec = self.eigvec_es[:, es_idx]

                intensity_x = abs(es_vec.conjugate() @ bigS_op_x @ gs_vec.T)**2
                intensity_y = abs(es_vec.conjugate() @ bigS_op_y @ gs_vec.T)**2
                intensity_z = abs(es_vec.conjugate() @ bigS_op_z @ gs_vec.T)**2
                intensities = (intensity_x, intensity_y, intensity_z)

                cross_intensity_matrix[es_idx][gs_idx] = intensities[axis_dict[axis]]
                cross_intensity_matrix_allpol[es_idx][gs_idx] = sum(intensities)
                cross_energies.append(((gs_idx, es_idx), (self.E_es[es_idx] - self.E_gs[gs_idx]).real))

                if intensities[axis_dict[axis]] > 10**-8:
                    usable_transitions.append((gs_idx, es_idx))

        # Next look at all possible transitions within ground states. Used for
        # the relaxation of states with the burnt state due to population depletion.
        # The matrix is symmetric so we only do on the upper half and just symmetrise.
        for gs_lower_idx in range(len(gs_blob)):
            gs_lower_vec = self.eigvec_gs[:, gs_blob[gs_lower_idx]]
            # Remember to do the diagonal (upper==lower)!
            for gs_upper_idx in range(gs_lower_idx, len(gs_blob)):
                gs_upper_vec = self.eigvec_gs[:, gs_blob[gs_upper_idx]]
                ground_intensity_matrix[gs_upper_idx][gs_lower_idx] = abs(gs_upper_vec.conjugate() @ bigS_op_axis @ gs_lower_vec.T)**2
                # Symmetrise the matrix
                ground_intensity_matrix[gs_lower_idx][gs_upper_idx] = ground_intensity_matrix[gs_upper_idx][gs_lower_idx]
        print("Done!")

        # Normalise by row (i.e. sum of transition targets from each state = 1)
        # Note that we have ignored the Zeeman states when considering the cross
        # and ground transitions but we assume that they are short-lived and thus
        # immediately decay. 
        cross_intensity_matrix /= np.sum(cross_intensity_matrix, axis=1, keepdims=True)
        cross_intensity_matrix_allpol /= np.sum(cross_intensity_matrix_allpol, axis=1, keepdims=True)
        ground_intensity_matrix /= np.sum(ground_intensity_matrix, axis=1, keepdims=True)

        cross_energies.sort(key=lambda x: x[1])
        cross_energies_keys = [transition[1] for transition in cross_energies]
        
        # Iterate through all pairs in burn_transitions that correspond to the 
        # inhomogeneous broadening.
        count = 1
        for gs_burn_idx, es_burn_idx in burn_transitions:
            print("Processing inhomogeneous pair {}/{}: ({}, {})".format(count, len(burn_transitions), gs_burn_idx, es_burn_idx))
            count += 1

            # gs_burn_idx, es_burn_idx is the principal levels that we are targeting
            # but there might be other transitions that are almost at the same 
            # energy that we are accidentally accessing. burn_transitions_list
            # will store that targetted + other nearby transitions.
            gs_burnt_set = set()
            es_burnt_set = set()
            burn_transitions_list = []

            # Energy of the principal holeburning pumping transition
            central_energy = (self.E_es[es_burn_idx] - self.E_gs[gs_burn_idx]).real
            # Look at all possible transitions and find those that are nearby
            # in transition energy.           
            i = bisect.bisect_left(cross_energies_keys, central_energy - 2*10**-3)
            j = bisect.bisect_right(cross_energies_keys, central_energy + 2*10**-3)

            for transition in cross_energies[i:j]:
                gs_burnt_set.add(transition[0][0])
                es_burnt_set.add(transition[0][1])

            ####################################################################
            # 1. COMPUTING GROUND STATE POPULATIONS
            ####################################################################
            # States that have high chance of transitioning to the burnt state
            # They will get burnt away too because the burnt state becomes empty
            additional_burnt_set = set()
            for gs_idx in gs_blob:
                for gs_burn_idx_iter in gs_burnt_set:
                    if ground_intensity_matrix[gs_idx][gs_burn_idx_iter] > 10**-5:
                        additional_burnt_set.add(gs_idx)

            gs_burnt_set = gs_burnt_set.union(additional_burnt_set)

            # Iterate through all levels in GS and find the ground state poulation
            # distribution by computing the probability of transitioning 
            # from es_burn_idx_iter. Here we only consider states in the correct Zeeman
            # branch and assume that any decays to the other Zeeman branch relax
            # down immediately. We exclude the gs_burnt_set states since they're
            # being burned away. We use allpol since internal relaxation can
            # be any polarisation.
            # Technically each es_burn_idx_iter is weighted by the number 
            # of ground states that are feeding towards that transition, but 
            # each excited state is fed by two ground states so we can just 
            # ignore this weighting and do an overall normalisation later.
            gs_populations = np.zeros(len(self.E_gs))

            for es_burn_idx_iter in es_burnt_set:
                for gs_idx in gs_blob:
                    if gs_idx in gs_burnt_set: 
                        continue
                    gs_populations[gs_idx] += cross_intensity_matrix_allpol[es_burn_idx_iter][gs_idx]

            # Normalise the population ratios to be equal to the number of 
            # burnt states. This makes sure that the total integral of the absorption
            # graph is zero.

            if np.sum(gs_populations) != 0: 
                gs_populations /= np.sum(gs_populations)
                gs_populations *= len(gs_burnt_set)
            else:
                print("Error in transition {}->{}".format(gs_burn_idx, es_burn_idx))       

            # Make some sort of relaxation in the GS for states that have strong coupling
            # As a first approximation we make them have equal population assuming
            # that the temperature is high enough to overcome the exp(-E/kT)
            for gs_idx in range(len(gs_populations)):
                for gs_idx_2 in range(gs_idx+1, len(gs_populations)):
                    if ground_intensity_matrix[gs_idx][gs_idx_2] > 10**-2:
                        avg = 0.5 * (gs_populations[gs_idx] + gs_populations[gs_idx_2])
                        gs_populations[gs_idx] = avg
                        gs_populations[gs_idx_2] = avg
  
            ####################################################################
            # 2. COMPUTING HOLES
            ####################################################################
            # Create the holes for transitions starting from gs_burnt_set 
            for gs_idx in gs_burnt_set:
                for es_idx in es_blob:
                    energy = (self.E_es[es_idx] - self.E_gs[gs_idx] - central_energy).real
                    intensity = cross_intensity_matrix[es_idx][gs_idx]
                    # Normalise with respect to the number of inhomog pairs we are looking at
                    intensity /= num_inhomog_atoms

                    if intensity > scaled_intensity_threshold:
                        spectrum_data.append((gs_idx, es_idx, energy, intensity))

            ####################################################################
            # 3. COMPUTING ANTIHOLES
            ####################################################################
            # Create the antiholes for usable transitions starting from the gs_idx
            # Usable means that the cross transition strength is above a threshold
            for gs_idx, es_idx in usable_transitions:
                # There are no antiholes from the burnt levels (would actually have  
                # been taken care off by the gs_populations list anyway)
                if gs_idx in gs_burnt_set or gs_populations[gs_idx] == 0: 
                    continue
                energy = (self.E_es[es_idx] - self.E_gs[gs_idx] - central_energy).real
                # Scale by both the absorption strength and the population size
                intensity = cross_intensity_matrix[es_idx][gs_idx] * gs_populations[gs_idx]
                # Normalise with respect to the number of inhomog pairs we are looking at
                intensity /= num_inhomog_atoms

                if intensity > scaled_intensity_threshold:
                    spectrum_data.append((gs_idx, es_idx, energy, -intensity))

        spectrum_data.sort()
        np.savetxt("SpectrumData_{}_{}_{}_{}_{}{}_peaks.txt".format(B, B_theta, B_phi, axis, len(self.neighbours), self.neighbours_string), 
            spectrum_data, 
            header="gs_idx, es_idx, energy, intensity",
            fmt=("%d", "%d", "%.4e", "%.4e"))

        return spectrum_data

    def plot_holeburning_spectrum(self, IN_FILE, FWHM, damp_factor, convolution, conv_width, linewidth):
        """ Given an in file with a list of transitions in the form 
        (gs_idx, es_idx, energy, intensity), plots the holeburning spectrum by
        placing a Lorentzian with a fixed FWHM at each peak. """

        # Read input values. Skip the 1st line which is just the column header.
        spectrum_data = np.loadtxt(IN_FILE, skiprows=1)
        # Read the IN_FILE name to get the params
        B, B_theta, B_phi = [float(x)  for x in IN_FILE.split("_")[1:4]]
        axis, label = IN_FILE.split("_")[4:6]

        # x-axis grid to compute the Lorentzian values
        xmin = -0.3
        xmax = 0.3
        E_grid = np.arange(xmin, xmax, FWHM/3)
        # We use this to store the predicted output holeburning spectrum
        holeburn_spectrum = np.zeros(len(E_grid))        

        # Iterate over each transition peak and compute the Lorentzian over the x-axis (E_grid)
        for row in spectrum_data:    
            lower_idx, upper_idx, energy, intensity = row
            holeburn_spectrum += (FWHM/2)**2 * intensity / ( (E_grid - energy)**2 + (FWHM/2)**2 )

            # Add dots to indicate the location of each peak (gets messy)
            # if abs(intensity) >= 0.005 and xmin < energy < xmax:
            #     if intensity > 0:
            #         plt.annotate('Hole ({}, {})'.format(lower_idx, upper_idx), xy=(energy, intensity))
            #         plt.plot([energy], [intensity], 'o', c='b')
            #     else:
            #         plt.annotate('Antihole ({}, {})'.format(lower_idx, upper_idx), xy=(energy, intensity))
            #         plt.plot([energy], [intensity], 'o', c='r')

        # Damp the entire spectrum by a certain factor
        holeburn_spectrum /= damp_factor

        if convolution:
            # Convolution of the spectrum by a rectangular/gaussian window
            window = np.array([1.0/6 if abs(E_grid[i]) < conv_width else 0 for i in range(len(E_grid))])
            # window = scipy.signal.gaussian(len(E_grid), 10) / 5
            plt.plot(E_grid, window)
            holeburn_spectrum = np.convolve(holeburn_spectrum, window, 'same')
            # holeburn_spectrum = scipy.signal.savgol_filter(holeburn_spectrum,31, 3)

        plt.plot(E_grid, holeburn_spectrum, label=label, linewidth=linewidth)
        plt.xlabel("Detuning / GHz")
        plt.ylabel("Transmittance")
        plt.title("Holeburning spectrum for (B, B_theta, B_phi) = {}, E along {}".format((B, B_theta, B_phi), axis))
        plt.xlim([xmin, xmax])

    def eval_holeburning_spectrum(self, x, FWHM, damp_factor, spectrum_data):
        """ Evaluates the holeburning spectrm at a certain x value instead of
        over the entire x-axis. Used for the numerical optimisation of the 
        values of FWHM and damping. """
        output = 0
        for row in spectrum_data:
            lower_idx, upper_idx, energy, intensity = row
            output += (FWHM/2)**2 * intensity / ( (x - energy)**2 + (FWHM/2)**2 )
        return output / damp_factor

    def plot_experimental_spectrum(IN_FILE):
        # Needs to in GHz
        # File format is (energy, transmittance_no_burn, tranmittance_with_burn)
        exp_data = np.loadtxt(IN_FILE, delimiter=",")
        xdata = exp_data[:, 0]
        ydata = np.log(exp_data[:, 2] / exp_data[:, 1])
        plt.plot(xdata, ydata, c='k', linewidth=1.5)

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
                       
yb171_neighbours_list = [
        # Generated from VESTA for YVO structure from a Y site to other Y's or V's
        # Sorted by distance
        # Format: ((cartesian displacement vector in Angstrom), spin, nuclear_g, atom type)
        
        # ((0.0, 0.0, -3.14465), 7/2, 1.4711, "V"), # 3.14465 
        # ((0.0, 0.0, 3.14465), 7/2, 1.4711, "V"),  # 3.14465

        # ((-3.55915, 0.0, 1.57232), 1/2, -0.2748308, "Y"), # 3.89098
        # ((3.55915, 0.0, 1.57232), 1/2, -0.2748308, "Y"), # 3.89098
        # ((0.0, -3.55915, -1.57232), 1/2, -0.2748308, "Y"), # 3.89098
        # ((0.0, 3.55915, -1.57232), 1/2, -0.2748308, "Y"), # 3.89098

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

yb171_electron_spin = 1/2
yb171_nuclear_spin = 1/2
yb171_g_nuclear = 0.98734

# Used to toggle the interactions on/off
electron_zeeman = 1
hyperfine = 1
nuclear_zeeman_rei = 1

A = SpinSystem(
    yb171_electron_spin,                                            # REI Electronic Spin
    yb171_nuclear_spin,                        # REI Nuclear Spin
    electron_zeeman * yb171_g_gs,                                   # REI Electronic GS g matrix
    electron_zeeman * yb171_g_es,                                   # REI ELectronic ES g matrix
    hyperfine * yb171_A_gs,                                         # GS Hyperfine A matrix
    hyperfine * yb171_A_es,                                         # ES Hyperfine A matrix
    nuclear_zeeman_rei * yb171_g_nuclear,                           # REI Nuclear g coeff
    yb171_neighbours_list
    )


################################################################################
# Number of superhyperfine multiplicity states
# mult = np.prod([neigh.multH for neigh in A.neighbours])
# A.compute_holeburning_spectrum(0.661, 0.33929, 0, "x", (range(mult), range(mult,2*mult)), (range(mult), range(mult,2*mult)))
# A.compute_holeburning_spectrum(1.0, 0.0, 0, "z", (range(mult),), (range(mult, 2*mult),))

# Read the experimental file and extract the x and y data series
# Take the ratio of burnt and unburnt and take log to get the optical depth. 
# Transmission = exp(-alpha*L) where alpha=absorption coeff which is proportional
# to population, and L is optical depth.

# exp_data = np.loadtxt("./HoleburningData/YbYVO_burnscan_B=1T_1.65K_holeburningexample/YbYVO_burnscan_B=1T_1.65K_holeburningexample.txt", delimiter=",")
# xdata = exp_data[:, 0]+0.172
# ybyvo_bbpit0MHZ_onres2_burn1_nice
# xdata = (exp_data[:, 0]-146.526)/1000 
# exp_data = np.loadtxt("./HoleburningData/ybyvo_bbpit0MHZ_onres2_burn1_nice.txt", delimiter=",")
# xdata = exp_data[:, 0]
# ydata = np.log(exp_data[:, 2] / exp_data[:, 1])

# Plot the predicted spectrum
# A.plot_holeburning_spectrum(IN_FILE="SpectrumData_0.661_0.33929_0_x_1V_peaks.txt", FWHM=0.000028, damp_factor=0.84, convolution=False, conv_width=0.0095, linewidth=2)

# Overlay the experimental data
# plt.plot(xdata, ydata, c='k', linewidth=1.5)
# plt.legend()

B_range = np.arange(0, 0.501, 0.001)
A.plot_transitions_strengths(B_range, np.pi/2, 0, 'optical', axis=None)
# plt.title()
plt.show()