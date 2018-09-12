import numpy as np
import copy
import math
import os
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["savefig.directory"] = "."
np.set_printoptions(precision=7, edgeitems=30, linewidth=100000)
mpl.rcParams.update({'font.size': 24})

class SpinSystem:

    mu_b = 9.2740100*10**-24 # Bohr Magneton (J/T)
    h = 6.62607004*10**-34   # Planck's constant (J s)
    beta_el = mu_b / (h * 10**9) # Bohr Magneton in GHz/T

    mu_n = 5.0507837*10**-27 # Nuclear Magneton (J/T)
    beta_n = mu_n / (h * 10**9) # Nuclear Magneton in GHz/T

    def generate_spin_matrices(self, spin):
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

    def unit_vector(self, theta, phi):
        return np.array([ np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),    
                          np.cos(theta)])

    def __init__(self, spinS, spinI, g_gs, g_es, A_gs, A_es, g_n_rei, g_n_host, spinH, disp_vector):
        # spinS = Electron spin, spinI = Nuclear spin, spinH = Nuclear spin of host atom
        self.spinS = spinS
        self.spinI = spinI
        self.spinH = spinH

        # gs = ground state, es = excited state
        # g = electronic zeeman coefficients (3x3 matrix)
        # g_n = REI Nuclear g coefficient (isotropic)
        # g_n_host = Nuclear magnetic moment coefficient of host atom
        self.g_gs = g_gs
        self.g_es = g_es
        self.g_n_rei = g_n_rei
        self.g_n_host = g_n_host

        # A = Hyperfine coefficients (3x3 matrix)
        self.A_gs = A_gs
        self.A_es = A_es
    
        # disp_vector = Displacement vector in angstroms to the host atom
        self.disp_vector = np.array(disp_vector)
        self.R = np.linalg.norm(self.disp_vector)

        # Multiplicities of the spin dimensions
        self.multS = int(2*self.spinS+1)
        self.multI = int(2*self.spinI+1)
        self.multH = int(2*self.spinH+1)

        # Populate the spin matrices 
        self.sx, self.sy, self.sz = self.generate_spin_matrices(self.spinS)
        # Concatenate them together. Dimensions: 3 x multS x multS
        self.S = np.array([self.sx, self.sy, self.sz])
        # Convert to a 3D vector with dimensions 3 x multS^2
        self.Sv3 = self.S.reshape((3, self.multS**2))

        # Do the same for the nuclear spin
        self.Ix, self.Iy, self.Iz = self.generate_spin_matrices(self.spinI)
        self.I = np.array([self.Ix, self.Iy, self.Iz])
        self.Iv3 = self.I.reshape((3, self.multI**2))

        # Do the same for the host nuclear spin
        self.Hx, self.Hy, self.Hz = self.generate_spin_matrices(self.spinH)
        self.H = np.array([self.Hx, self.Hy, self.Hz])
        self.Hv3 = self.H.reshape((3, self.multH**2))

        # Hamiltonian currently consists of only the hyperfine and SHF pieces 
        # since they are independent of the B field.
        self.initialize_HHF()
        self.initialize_SHF()
        self.reset_ham()

        # Update list of energy levels for each state
        self.calc_energies()

        # self.neighbours = 10**-10 * np.array([
        # # Generated from VESTA for YVO structure from Y to other Y's or V's
        # (0.0, 0.0, -3.14465), # V, 3.14465
        # (0.0, 0.0, 3.14465), # V, 3.14465

        # (-3.55915, 0.0, 1.57232), # Y, 3.89098
        # (3.55915, 0.0, 1.57232), # Y, 3.89098
        # (0.0, -3.55915, -1.57232), # Y, 3.89098
        # (0.0, 3.55915, -1.57232), # Y, 3.89098

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
        # ])

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
        self.HHF_gs = (self.A_gs @ self.Sv3).reshape((3, self.multS, self.multS))
        self.HHF_gs = sum(np.kron(self.I[i], self.HHF_gs[i]) for i in range(3))
        # Expand into the host nucleus space
        self.HHF_gs = np.kron(self.HHF_gs, np.identity(self.multH))

        # Repeat for the excited state
        self.HHF_es = (self.A_es @ self.Sv3).reshape((3, self.multS, self.multS))
        self.HHF_es = sum(np.kron(self.I[i], self.HHF_es[i]) for i in range(3))
        # Expand into the host nucleus space
        self.HHF_es = np.kron(self.HHF_es, np.identity(self.multH))

    def initialize_SHF(self):
        # mu = mu_b * g * S
        # Use v3 so that we can take the product with the 3x3 g matrix
        # Then reshape so that we get back our column 3 x (multS x multS) 
        mu_REI_gs = -(self.mu_b * self.g_gs @ self.Sv3).reshape((3, self.multS, self.multS)) 
        # TODO: Should I add a negative sign to the electron mu?
        mu_REI_es = -(self.mu_b * self.g_es @ self.Sv3).reshape((3, self.multS, self.multS))
        # No need for v3 for host since we assume an isotropic g so we only 
        # need scalar multiplication. Note mu_n instead of mu_b!
        mu_host = self.mu_n * self.g_n_host * self.H

        # First term is (mu_REI)dot(mu_host)/R^3
        first_term_gs = sum(np.kron(np.kron(mu_REI_gs[i], np.identity(self.multI)), mu_host[i]) for i in range(3)) / self.R**3
        first_term_es = sum(np.kron(np.kron(mu_REI_es[i], np.identity(self.multI)), mu_host[i]) for i in range(3)) / self.R**3

        # Reshape both back to v3 since we need another dot product, this time with R
        mu_REI_gs_v3 = mu_REI_gs.reshape((3, self.multS**2))
        mu_REI_es_v3 = mu_REI_es.reshape((3, self.multS**2))
        mu_host_v3 = mu_host.reshape((3, self.multH**2))
        # Take the dot product then reshape back
        dot_REI_gs = self.disp_vector.dot(mu_REI_gs_v3).reshape((self.multS, self.multS))
        dot_REI_es = self.disp_vector.dot(mu_REI_es_v3).reshape((self.multS, self.multS))
        dot_host = self.disp_vector.dot(mu_host_v3).reshape((self.multH, self.multH))

        # Second term is 3(mu_REI dot R)(mu_host dot R)/R^5
        second_term_gs = (3/self.R**5) * np.kron(np.kron(dot_REI_gs, np.identity(self.multI)), dot_host)
        second_term_es = (3/self.R**5) * np.kron(np.kron(dot_REI_es, np.identity(self.multI)), dot_host)

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
        HZ_el_gs = np.kron(HZ_el_gs, np.identity(self.multH))
        HZ_el_es = np.kron(HZ_el_es, np.identity(self.multH))

        # Do the same for the REI nuclear Zeeman Hamiltonian but there is no 
        # distinction between GS and ES. We assume the nuclear g factor is an 
        # isotropic scalar.
        HZ_n_rei = (self.beta_n * self.g_n_rei * B_vec @ self.Iv3).reshape((self.multI, self.multI))
        # Expand into the electronic spin space
        HZ_n_rei = np.kron(np.identity(self.multS), HZ_n_rei)
        # Expand into the superhyperfine nuclear spin space
        HZ_n_rei = np.kron(HZ_n_rei, np.identity(self.multH))

        # Do the same for the nuclear Zeeman Hamiltonian  for the host nuclear spin
        HZ_n_host = (self.beta_n * self.g_n_host * B_vec @ self.Hv3).reshape((self.multH, self.multH))
        # Expand into the REI nuclear spin space
        HZ_n_host = np.kron(np.identity(self.multI), HZ_n_host)
        # Expand into the electronic spin space
        HZ_n_host = np.kron(np.identity(self.multS), HZ_n_host)

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

    def optical_transitions(self, B=None, B_theta=None, B_phi=None):
        """ Computes the transitions between the energy levels. Optional to
        provide a B field to update before computing. """

        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        transitions = []

        for idx_g, e_g in enumerate(self.E_gs):
            for idx_e, e_e in enumerate(self.E_es):
                transitions.append((e_e-e_g, self.transition_strength(self.eigvec_gs[:, idx_g], self.eigvec_es[:, idx_e])))
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


        # Plain inner product overlap
        # r_23 = abs(gs_upper.conjugate() @ es_lower.T)**2
        # r_13 = abs(gs_lower.conjugate() @ es_lower.T)**2
        # Overlap while enclosing the Sx, Sy, Sz matrices
        r_23_S_x = abs(gs_upper.conjugate() @ bigSx @ es_lower.T)**2
        r_13_S_x = abs(gs_lower.conjugate() @ bigSx @ es_lower.T)**2
        # r_23_S_y = abs(gs_upper.conjugate() @ bigSy @ es_lower.T)**2
        # r_13_S_y = abs(gs_lower.conjugate() @ bigSy @ es_lower.T)**2
        # r_23_S_z = abs(gs_upper.conjugate() @ bigSz @ es_lower.T)**2
        # r_13_S_z = abs(gs_lower.conjugate() @ bigSz @ es_lower.T)**2

        # Various ratios
        # R = r_23 / r_13
        RS_x = r_23_S_x / r_13_S_x
        # RS_y = r_23_S_y / r_13_S_y
        # RS_z = r_23_S_z / r_13_S_z
        # Various rhos
        # rho = 4 * R / (1 + R)**2
        rho_S_x = 4 * RS_x / (1 + RS_x)**2
        # rho_S_y = 4 * RS_y / (1 + RS_y)**2
        # rho_S_z = 4 * RS_z / (1 + RS_z)**2

        # print("Compare!")
        # print("R={}, RS={}, Diff%={}, B={}, B_theta={}, B_phi={}".format(R, RS, (RS-R)/RS*100, B, B_theta, B_phi))

        # return r_23, r_13, R, rho
        return r_23_S_x, r_13_S_x, RS_x, rho_S_x, gs_lower, gs_upper, es_lower
        # return r_23_S_y, r_13_S_y, RS_y, rho_S_y
        # return r_23_S_z, r_13_S_z, RS_z, rho_S_z
        # return max(r_23_S_x, r_23_S_y, r_23_S_z), max(r_13_S_x, r_13_S_y, r_13_S_z), max(RS_x, RS_y, RS_z), max(rho_S_x, rho_S_y, rho_S_z)

    def max_branching_contrast(self, B_range, B_theta, B_phi, remove_biggest):
        """ Looks over a certain range of B fields and try to find the maximum
        value that the branching contrast rho takes for the current displacement
        vector in the self object. Has an option to remove the largest point to 
        remove any spurious points from B=0. """

        # The [3] is to take the rho part of the output from the branching_contrast
        # outoput. [-1] and [-2] take the largest and 2nd largest rho respectively.
        if remove_biggest:
            # return sorted([self.branching_contrast(b, B_theta, B_phi)[3] for b in B_range])[-2]
            return sorted([self.branching_contrast(b, B_theta, B_phi)[3] for b in B_range])[-2]
        else:
            return sorted([self.branching_contrast(b, B_theta, B_phi)[3] for b in B_range])[-1]

    # def max_overlap_rho(self, B_range, B_theta, B_phi, remove_biggest):
    #     """ Looks over a certain range of B fields and try to find the maximum
    #     value that the product of the overlap |<2|3>|^2 and  rho takes for the 
    #     current displacement vector in the self object. Has an option to remove 
    #     the largest point to remove any spurious points from B=0. """

    #     # The [3] is to take the rho part of the output from the branching_contrast
    #     # output. [-1] and [-2] take the largest and 2nd largest rho respectively.
    #     sweep_list = [self.branching_contrast(b, B_theta, B_phi) for b in B_range]
    #     overlap_rho_list = sorted([x[0] * x[3] for x in sweep_list])

    #     if remove_biggest:
    #         return overlap_rho_list[-2]
    #     else:
    #         return overlap_rho_list[-1]

    def max_transition_overlap(self, B_range, B_theta, B_phi, remove_biggest):
        """ Looks over a certain range of B fields and try to find the maximum
        value that the transition overlap <f|i> takes for the current displacement
        vector in the self object. Has an option to remove the largest point to 
        remove any spurious points from B=0. """

        # The [0] is to take the |<2|3>|^2 part of the output from the branching_contrast
        # output. [-1] and [-2] take the largest and 2nd largest rho respectively.
        if remove_biggest:
            return sorted([self.branching_contrast(b, B_theta, B_phi)[0] for b in B_range])[-2]
        else:
            return sorted([self.branching_contrast(b, B_theta, B_phi)[0] for b in B_range])[-1]

    def transition_strength(self, initial, final):
        bigSx = 2*np.kron(np.kron(self.sx, np.identity(self.multI)), np.identity(self.multH))
        bigSy = 2*np.kron(np.kron(self.sy, np.identity(self.multI)), np.identity(self.multH))
        bigSz = 2*np.kron(np.kron(self.sz, np.identity(self.multI)), np.identity(self.multH))
        
        # TODO: Find out why the nuclear spin doesn't matter?
        # bigSx = 4*np.kron(np.kron(self.sx, self.Ix), np.identity(self.multH))
        # bigSy = 4*np.kron(np.kron(self.sy, self.Iy), np.identity(self.multH))
        # bigSz = 4*np.kron(np.kron(self.sz, self.Iz), np.identity(self.multH))

        # bigSx = 8*np.kron(np.kron(self.sx, self.Ix), self.Hx)
        # bigSy = 8*np.kron(np.kron(self.sy, self.Iy), self.Hy)
        # bigSz = 8*np.kron(np.kron(self.sz, self.Iz), self.Hz)

        fx = abs(final.conjugate() @ bigSx @ initial) ** 2
        fy = abs(final.conjugate() @ bigSy @ initial) ** 2
        fz = abs(final.conjugate() @ bigSz @ initial) ** 2
        return fx, fy, fz

    def neighbours_max_overlap(self, B_theta, B_phi, search_all):
        max_params_list = []

        # Loop over all the nearest neighbours and set that neighbour to be the
        # interaction partner.
        for idx, neigh in enumerate(self.neighbours):
            
            if search_all:
                print("Neighbour {}/{}".format(idx+1, len(self.neighbours)))
            else:
                print("Neighbour {}/{}\r".format(idx+1, len(self.neighbours)), end="")

            self.set_vector(neigh)
            # Dynamically scale the B search range based on the vector length
            B_range = np.arange(0.15, 31, 0.15) / ((self.R / 10**-10) ** 3)
            B_states_sweep = []

            # Sweep over the given range of B fields AND the number of possible 
            # states combination and get the branching contrast parameters: 
            # |<2|3>|^2, |<1|3>|^2, R, rho
            if search_all:
                counter = 1
                for i in range(self.eigvec_gs.shape[1]):
                    for j in range(i+1, self.eigvec_gs.shape[1]):
                        for k in range(self.eigvec_es.shape[1]):
                            print("Iterating state combination {}/{}\r".format(counter, self.eigvec_gs.shape[1] * (self.eigvec_gs.shape[1]-1) // 2 * self.eigvec_es.shape[1]), end="")
                            counter += 1

                            for B in B_range:
                                B_states_sweep.append([B, (i, j, k), self.branching_contrast(B, B_theta, B_phi, (i, j, k))])
                print()
            else:
                for B in B_range:
                    B_states_sweep.append([B, (0, 1, 0), self.branching_contrast(B, B_theta, B_phi, (0, 1, 0))])

            # Sort by the 1st entry which is |<2|3>|^2 times the 4th entry which is rho
            B_states_sweep.sort(key=lambda x: x[2][0] * x[2][3])
            # Extract the largest params
            max_params = B_states_sweep[-1]

            # Each entry contains the atom's position, the B field for max overlap,
            # then the 7-tuple overlap parameters. This is the best for a particular
            # atom after sweeping over B field intensity and states.
            max_params_list.append([neigh] + max_params)

        # Previously we sorted each **indiv B sweep** for a point to get the B mag
        # that gave the biggest overlap times rho. 
        # Now we sort among all **points** to get the point that has the maximum 
        # overlap times rho.
        max_params_list.sort(key=lambda x: x[3][0] * x[3][3])
        return max_params_list[-1]

    ############################################################################
    
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
                transitions = self.optical_transitions(B, B_theta, B_phi)
            elif transition_type == "spin_gs":
                transitions = self.spin_transitions(B, B_theta, B_phi)[0]
            elif transition_type == "spin_es":
                transitions = self.spin_transitions(B, B_theta, B_phi)[1]
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
        E_grid = np.arange(-10, 10, 0.001)
        E_grid_size = len(E_grid)
        opt_trans = self.optical_transitions(B, B_theta, B_phi)
        
        if axis is None:
            # Used to represent each column in the grid which we add each peak onto
            optical_peaks_x, optical_peaks_y, optical_peaks_z = np.zeros(E_grid_size), np.zeros(E_grid_size), np.zeros(E_grid_size)
        else:
            optical_peaks = np.zeros(E_grid_size)

        # For each possible optical transition at the particular value of 
        # B field, we take the transition energy (0-th comp) and create a 
        # Lorentzian peak around in the E_grid space, then scale it by its
        # amplitude given by transition[1] (x,y,z) comps.
        for transition in opt_trans:
            # FWHM of the Lorentzian
            fwhm = 0.3
            if axis is None:
                optical_peaks_x += (fwhm/2)**2 * transition[1][0] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                optical_peaks_y += (fwhm/2)**2 * transition[1][1] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
                optical_peaks_z += (fwhm/2)**2 * transition[1][2] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
            else:
                optical_peaks += (fwhm/2)**2 * transition[1][axis] / ( (E_grid - transition[0].real)**2 + (fwhm/2)**2 )
            
        # Plot a slice of the intensity at a particular value of B field
        plt.figure()
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

    def plot_neighbours_max_overlaprho_Bmap(self, theta_grid_size=None, phi_grid_size=None, file_in=None, search_all=False):

        if theta_grid_size is None:
            theta_grid_size = 9
        if phi_grid_size is None:
            phi_grid_size = 9

        grid = np.zeros((phi_grid_size, theta_grid_size))
        B_phis = np.linspace(0, 2*np.pi, phi_grid_size)
        B_thetas = np.linspace(0, np.pi, theta_grid_size)

        # Holds the current max overlap amplitude
        current_max = 0
        current_max_params = None

        if search_all:
            data_filename = "YbNeighbourOverlapRho_SearchAll_BMap_{0}x{1}.txt".format(theta_grid_size, phi_grid_size)
            max_params_filename = "YbNeighbourOverlapRho_SearchAll_BMap_{0}x{1}_maxparams.txt".format(theta_grid_size, phi_grid_size)
        else:
            data_filename = "YbNeighbourOverlapRho_BMap_{0}x{1}.txt".format(theta_grid_size, phi_grid_size)
            max_params_filename = "YbNeighbourOverlapRho_BMap_{0}x{1}_maxparams.txt".format(theta_grid_size, phi_grid_size)

        if file_in is None:
            for idx_phi, B_phi in enumerate(B_phis):
                for idx_theta, B_theta in enumerate(B_thetas):

                    print("Grid position: {}/{}".format(1+idx_theta+theta_grid_size*idx_phi, theta_grid_size*phi_grid_size)) # Progress bar

                    # Get the (vector, B, (|<2|3>|^2, |<1|3>|^2, R, rho)) for the 
                    # neighbour that has the largest |<2|3>|^2 * rho
                    # This already iterates through all the neighbours
                    max_overlap_params = self.neighbours_max_overlap(B_theta, B_phi, search_all)

                    # Used for finding the best B field orientation across
                    # the entire map.
                    if max_overlap_params[3][0] * max_overlap_params[3][3] > current_max:
                        current_max = max_overlap_params[3][0] * max_overlap_params[3][3]
                        current_max_params = (B_theta, B_phi, max_overlap_params)

                    # Plots the best *neighbour* for the particular B direction
                    grid[idx_phi][idx_theta] =  max_overlap_params[3][0] * max_overlap_params[3][3]

            # We output the grid of best neighbour outcome as a function of B 
            # field orientation.
            np.savetxt(data_filename, grid)
            
            # We output the single best neighbour over all possible B field oreintations.
            print(current_max_params)
            B_theta, B_phi, max_overlap_params = current_max_params
            print(B_theta)
            neigh_vec, max_B, state_indices, max_overlap_tuple = max_overlap_params
            # Star means variable length tuple unpacking
            *max_over_tuple_nums, gs_lower, gs_upper, es_lower = max_overlap_tuple

            header = "B_theta, B_phi, neigh_vec (x,y,z), max_B, state_indices, overlap_4tuple, gs_lower, gs_upper, es_lower"
            np.savetxt(max_params_filename, [B_theta, B_phi] + list(neigh_vec) + 
                [max_B] + list(state_indices) + list(max_over_tuple_nums) +
                list(gs_lower) + list(gs_upper) + list(es_lower), header=header)

        else:
            grid = np.loadtxt(file_in)
            B_phis = np.linspace(0, 2*np.pi, phi_grid_size)
            B_thetas = np.linspace(0, np.pi, theta_grid_size)

        # Theta = x-axis, Phi = y-axis
        plt.figure()
        plt.contourf(B_thetas, B_phis, grid, np.unique(grid))
        plt.xlabel("B_Theta")
        plt.ylabel("B_Phi")
        plt.colorbar()

        plt.figure()
        plt.imshow(grid)
        plt.xlabel("B_Theta")
        plt.ylabel("B_Phi")
        plt.colorbar()

    def plot_fn_map(self, B_theta, B_phi, map_fn, filename, grid_size, file_in):
        """ Plot the theta-phi map of the maximum branching contrast rho (as the B
        field is varied). """ 
        
        if file_in is None:
            # Size of grid in theta-phi space we want to iterate over
            if grid_size is None:
                grid_size = 25

            grid = np.zeros((grid_size, grid_size))
            phis = np.linspace(0, 2*np.pi, grid_size)
            thetas = np.linspace(0, np.pi, grid_size)

            # Range of B fields to search over to find the max param
            # This range depends on the length of the vector used below!
            # This range is ok for for prefactor 10**-10
            B_range = np.arange(0, 20, 0.1)

            # Iterate over the grid. Theta = x-axis, Phi = y-axis
            for idx_phi, phi in enumerate(phis):
                print("{}/{}".format(idx_phi, grid_size)) # Progress bar

                for idx_theta, theta in enumerate(thetas):
                    vector = 10**-10 * self.unit_vector(theta, phi)
                    self.set_vector(vector)
                    # Find the max branching contrast at that angle with B field at a  
                    # certain angle then store in the array. 
                    # We remove the biggest to avoid any problems with 0 field.
                    grid[idx_phi][idx_theta] =  map_fn(B_range, B_theta, B_phi, remove_biggest=True)

            np.savetxt(filename, grid)

        else:
            grid = np.loadtxt(file_in)
            grid_size = len(grid)
            phis = np.linspace(0, 2*np.pi, grid_size)
            thetas = np.linspace(0, np.pi, grid_size)

        # Theta = x-axis, Phi = y-axis
        plt.figure()
        plt.contourf(thetas, phis, grid, 100)
        plt.xlabel("Theta")
        plt.ylabel("Phi")
        plt.colorbar()

    def plot_max_branching_contrast_map(self, B_theta, B_phi, grid_size=None, file_in=None):
        """ Plot the theta-phi map of the maximum branching contrast rho (as the B
        field is varied). """ 

        self.plot_fn_map(B_theta, B_phi, 
            self.max_branching_contrast, 
            "YbBranchingContrastMap_{0}x{0}_{1}_{2}.txt".format(grid_size, int(B_theta/np.pi*180), int(B_phi/np.pi*180)),
            grid_size, file_in)

    # def plot_max_overlap_rho_map(self, B_theta, B_phi, grid_size=None, file_in=None):
    #     """ Plot the theta-phi map of the maximum product of overlap and rho (as the B
    #     field is varied). """ 

    #     self.plot_fn_map(B_theta, B_phi, 
    #         self.max_overlap_rho, 
    #         "YbOverlapRhoMap_{0}x{0}_{1}_{2}.txt".format(grid_size, int(B_theta/np.pi*180), int(B_phi/np.pi*180)),
    #         grid_size, file_in)

    def plot_max_transition_overlap_map(self, B_theta, B_phi, grid_size=None, file_in=None):
        """ Plot the theta-phi map of the maximum transition amplitude |<2|3>|^2 (as the B
        field is varied). """ 

        self.plot_fn_map(B_theta, B_phi, 
            self.max_transition_overlap, 
            "YbTransitionOverlapMap_{0}x{0}_{1}_{2}.txt".format(grid_size, int(B_theta/np.pi*180), int(B_phi/np.pi*180)),
            grid_size, file_in)
        
    def overlay_atoms(self):
        """ Overlay the position of atoms onto the theta-phi map. Takes the 
        Cartesian coordinates of atoms and converts them into angular coordinatess.
        To be used with the plot_max_branching_contrast_map function. """

        # Compute phi from the x,y coords
        neigh_phi = [math.atan2(coord[1], coord[0]) for coord in self.neighbours]
        # Just to make things positive
        neigh_phi = [phi if phi>0 else phi+2*np.pi for phi in neigh_phi]

        neigh_theta = [math.acos(coord[2]/np.linalg.norm(np.array(coord))) for coord in self.neighbours]

        # All atoms are plotted red. Each colour information is encoded as an RGBA tuple.
        colours = np.zeros((len(self.neighbours), 4))
        colours[:, 0] = 1
        # Scale all distances by the maximum distace, then invert the scale
        # since alpha=0 means transparent. Set the minimum alpha to be 0.15
        # and the maximum alpha to be 1.
        if len(self.neighbours) > 1:
            distances = np.array([np.linalg.norm(neigh) for neigh in self.neighbours])
            distances = 1 - 0.85 * (distances - distances[0]) / (distances[-1] - distances[0])
        else:
            distances = 1

        colours[:, 3] = distances

        plt.scatter(neigh_theta, neigh_phi, c=colours)

    def plot_branching_contrast_map_brange(self, B_range, B_theta, B_phi):
        """ Plots a theta-phi map in angular space of the branching contrast rho.
        However, instead of finding the maximum possible value of rho for that
        point, we have a 3x3 grid of various B fields, and then we just compute 
        the value of rho for that particular value of B. """ 
        
        plt.figure()
        grid_size = 50
        phis = np.linspace(0, 2*np.pi, grid_size)
        thetas = np.linspace(0, np.pi, grid_size)
        assert(len(B_range)==9)

        for i in range(1, 10):
            branching_contrast_map = np.zeros((grid_size, grid_size))
            for idx_phi, phi in enumerate(phis):
                for idx_theta, theta in enumerate(thetas):
                    vector = 10**-10 * self.unit_vector(theta, phi)
                    self.set_vector(vector)
                    branching_contrast_map[idx_phi][idx_theta] =  self.branching_contrast(B_range[i-1], B_theta, B_phi)[3]

            plt.subplot("33"+str(i))
            plt.contourf(thetas, phis, branching_contrast_map, 100, vmin=0, vmax=1)
            plt.xlabel("Theta")
            plt.ylabel("Phi")
            plt.colorbar()

    def plot_branching_contrast_angular(self, theta, phi, B_range, B_theta, B_phi):
        """ Plots the branching contrast rho as a function of B for a particualr 
        fixed point in theta-phi angular space. """
        vector = 10**-10  * self.unit_vector(theta, phi)
        self.set_vector(vector)
        plt.figure()
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[0] for b in B_range], 'ro', label="23")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[1] for b in B_range], 'bo', label="13")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[2] for b in B_range], 'k^', label="R")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[3] for b in B_range], 'gv', label="rho")
        plt.ylim(0, 1)
        plt.legend()

    def plot_branching_contrast_cartesian(self, cartesian_vector, B_range, B_theta, B_phi):
        """ Plots the branching contrast rho as a function of B for a particualr 
        fixed point in theta-phi angular space. """
        self.set_vector(cartesian_vector)
        plt.figure()
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[0] for b in B_range], 'ro', label="23")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[1] for b in B_range], 'bo', label="13")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[2] for b in B_range], 'k^', label="R")
        plt.plot(B_range, [self.branching_contrast(b, B_theta, B_phi)[3] for b in B_range], 'gv', label="rho")
        plt.ylim(0, 1)
        plt.legend()

    def plot_eigenvector_comps(self, B_range, B_theta, B_phi):
        gs_lower_list = np.zeros((len(B_range), 8))
        gs_upper_list = np.zeros((len(B_range), 8))
        es_lower_list = np.zeros((len(B_range), 8))
        es_upper_list = np.zeros((len(B_range), 8))
        e_gs = np.zeros((len(B_range), 2))
        e_es = np.zeros((len(B_range), 2))

        for idx, b in enumerate(B_range):
            self.update_B(b, B_theta, B_phi)

            eigval_gs, eigvec_gs = np.linalg.eig(self.H_gs)
            # Sort the columns of the eigvecs in increasing size of the eigvals
            eigvec_gs = eigvec_gs[:, np.argsort(eigval_gs)]

            gs_lower_list[idx, :4] = eigvec_gs[:, 0].real
            gs_lower_list[idx, 4:] = eigvec_gs[:, 0].imag
            gs_upper_list[idx, :4] = eigvec_gs[:, 1].real
            gs_upper_list[idx, 4:] = eigvec_gs[:, 1].imag
         
            # Do the same for the excited state
            eigval_es, eigvec_es = np.linalg.eig(self.H_es)
            # Sort the columns of the eigvecs in increasing size of the eigvals
            eigvec_es = eigvec_es[:, np.argsort(eigval_es)]

            es_lower_list[idx, :4] = eigvec_es[:, 0].real
            es_lower_list[idx, 4:] = eigvec_es[:, 0].imag
            es_upper_list[idx, :4] = eigvec_es[:, 1].real
            es_upper_list[idx, 4:] = eigvec_es[:, 1].imag

        plt.figure()
        plt.suptitle("Eigenvectors for the lowest 2 energy eigenvectors for the ground and excited state (4 comps)")
        plt.subplot(221)
        plt.plot(B_range, gs_lower_list)
        plt.title("Lowest ground state eigenvector")
        plt.legend(["Re(1)", "Re(2)", "Re(3)", "Re(4)", "Im(1)", "Im(2)", "Im(3)", "Im(4)"], loc=1)
        plt.subplot(222)
        plt.plot(B_range, gs_upper_list)
        plt.title("2nd lowest ground state eigenvector")
        plt.legend(["Re(1)", "Re(2)", "Re(3)", "Re(4)", "Im(1)", "Im(2)", "Im(3)", "Im(4)"], loc=1)
        plt.subplot(223)
        plt.plot(B_range, es_lower_list)
        plt.title("Lowest excited state eigenvector")
        plt.legend(["Re(1)", "Re(2)", "Re(3)", "Re(4)", "Im(1)", "Im(2)", "Im(3)", "Im(4)"], loc=1)
        plt.subplot(224)
        plt.plot(B_range, es_upper_list)
        plt.title("2nd lowest excited state eigenvector")
        plt.legend(["Re(1)", "Re(2)", "Re(3)", "Re(4)", "Im(1)", "Im(2)", "Im(3)", "Im(4)"], loc=1)

        self.plot_superhyperfine(B_range, B_theta, B_phi)

    def plot_holeburning_spectrum(self, B, B_theta, B_phi, holeburn_transition, axis):

        # Update the B field 
        self.update_B(B, B_theta, B_phi)
        # Unpack the lower (from GS) and upper (from ES) level for the pumping
        lower_burn, upper_burn = holeburn_transition

        # Set the axis for the E field polarisation to use the appropriate S matrix
        # for the transition probabilities
        S_op = {0: self.sx, 1 : self.sy, 2 : self.sz}
        bigS_op = 2*np.kron(np.kron(S_op[axis], np.identity(self.multI)), np.identity(self.multH))

        # Used to store the populations in the GS that fall from upper_burn
        gs_populations = np.zeros(len(self.E_gs))
        # Iterate through all levels in GS and find prob of transitioning from upper_burn
        # We exclude the lower_burn state since it's being burned away
        upper_vec = self.eigvec_es[:, upper_burn]
        for lower_idx in range(len(self.E_gs)):
            if lower_idx == lower_burn:
                continue
            lower_vec = self.eigvec_gs[:, lower_idx]
            gs_populations[lower_idx] = abs(upper_vec.conjugate() @ bigS_op @ lower_vec.T)**2

        # Normalise the population ratios to be 1
        gs_populations /= np.sum(gs_populations)

        # plt.plot(gs_populations, 'o-')
        # plt.xlabel("Ground state index")
        # plt.ylabel("Excess population after holeburning")
        axis_dict = {0: "x", 1: "y", 2: "z"}
        # plt.title("Ground state population for (B, B_theta, B_phi) = {} with transition {}, E along {}".format((B, B_theta, B_phi), holeburn_transition, axis_dict[axis]))

        # plt.figure()

        # We use this to store the predicted output holeburning spectrum
        E_grid = np.arange(-0.2, 0.2, 0.00001)
        holeburn_spectrum = np.zeros(len(E_grid))
        # Each peak's FWHM
        FWHM = 0.00001
        # Energy of the holeburning pumping transition
        central_energy = self.E_es[upper_burn] - self.E_gs[lower_burn]

        # First create the holes for transitions starting from the lower_burn
        lower_vec = self.eigvec_gs[:, lower_burn]
        for upper_idx in range(len(self.E_es)):
            upper_vec = self.eigvec_es[:, upper_idx]

            energy = (self.E_es[upper_idx] - self.E_gs[lower_burn] - central_energy).real
            intensity = abs(upper_vec.conjugate() @ bigS_op @ lower_vec.T)**2

            holeburn_spectrum += (FWHM/2)**2 * intensity / ( (E_grid - energy.real)**2 + (FWHM/2)**2 )

            if intensity >= 0.025:
                plt.annotate('Hole ({}, {})'.format(lower_burn, upper_idx), xy=(energy, intensity))
                plt.plot([energy], [intensity], 'o', c='b')

        # Next create the antiholes for transitions starting from the lower_idx
        for lower_idx in range(len(self.E_gs)):
            if lower_idx == lower_burn: 
                continue
            for upper_idx in range(len(self.E_es)):
                lower_vec = self.eigvec_gs[:, lower_idx]
                upper_vec = self.eigvec_es[:, upper_idx]
                energy = (self.E_es[upper_idx] - self.E_gs[lower_idx] - central_energy).real
                intensity = abs(upper_vec.conjugate() @ bigS_op @ lower_vec.T)**2 * gs_populations[lower_idx]

                holeburn_spectrum -= (FWHM/2)**2 * intensity / ( (E_grid - energy.real)**2 + (FWHM/2)**2 )

                if intensity >= 0.025:
                    plt.annotate('Antihole ({}, {})'.format(lower_idx, upper_idx), xy=(energy, -intensity))
                    plt.plot([energy], [-intensity], 'o', c='r')

        plt.plot(E_grid, holeburn_spectrum, c='k')
        plt.xlabel("Detuning / GHz")
        plt.ylabel("Transmittance")
        plt.title("Holeburning spectrum for (B, B_theta, B_phi) = {} with transition {}, E along {}".format((B, B_theta, B_phi), holeburn_transition, axis_dict[axis]))
        plt.xlim([-0.2, 0.2])

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

yb173_A_gs = np.array([[-0.186, 0,  0],
                       [0, -0.186,   0],
                       [0, 0,  1.328]])


# yb173_A_es = np.array([[-0.959,   0,  0],
#                        [0, -0.959,  0],
#                        [0,   0,  -1.276]])

# yb173_g_gs = np.array([[0.85, 0,  0],
#                        [0,  0.85,  0],
#                        [0,  0, -6.08]])

# yb173_g_es = np.array([[1.6, 0,  0],
#                        [0, 1.6,  0],
#                        [0,  0, 2.51]])

# er_g_gs =   np.array([[3.070, -3.124,  3.396],
#                        [-3.124,  8.156,  -5.756],
#                          [3.396,  -5.756, 5.787]])

# er_g_es =   np.array([[1.950, -2.212, 3.584],
#                          [-2.212, 4.232,  -4.986],
#                          [3.584,  -4.986, 7.888]])

# Parallel = z, perp = xy
# nd_g_gs = np.diag((-2.361, -2.361, -0.915))
# nd_g_es = np.diag((-0.28, -0.28, -1.13))
    
electron_zeeman = 1
hyperfine = 1
nuclear_zeeman_rei = 1
nuclear_zeeman_host = 1
superhyperfine = 1

A = SpinSystem(
    1/2,                                 # Yb-171                   # REI Electronic Spin
    nuclear_zeeman_rei * 1/2,            # Yb-171                   # REI Nuclear Spin
    electron_zeeman * yb171_g_gs,                                   # REI Electronic GS g matrix
    electron_zeeman * yb171_g_es,                                   # REI ELectronic ES g matrix
    hyperfine * yb171_A_gs,                                         # GS Hyperfine A matrix
    hyperfine * yb171_A_es,                                         # ES Hyperfine A matrix
    nuclear_zeeman_rei * 0.98734,       # Yb-171                  # REI Nuclear g coeff
    nuclear_zeeman_host * -0.2748308,   # Y-89                    # Host Nuclear g coeff
    1/2,                                 # Y-89                     # Host Nuclear spin
    superhyperfine * 10**-10 * np.array((-3.55915, 0.0, 1.57232))  # Displacement vector to host
    )

#########################################
# (0.0, 0.0, -3.14465), # V, 3.14465
# (0.0, 0.0, 3.14465), # V, 3.14465
#########################################
# (-3.55915, 0.0, 1.57232), # Y, 3.89098
# (3.55915, 0.0, 1.57232), # Y, 3.89098
# (0.0, -3.55915, -1.57232), # Y, 3.89098
# (0.0, 3.55915, -1.57232), # Y, 3.89098
#########################################

# 7.120, 6.289
# Nuclear Spin: 7/2 (V-51), 1/2 (Y-89), 1/2 (Yb-171), 5/2 (Yb-173)

################################################################################

A.plot_energies_brange(np.arange(0, 0.201, 0.001), 0*np.pi/2, 0)
# A.plot_spin_transitions(np.arange(0, 0.301, 0.001), np.pi/4, 0)
# A.plot_transitions_strengths(np.arange(0,0.301,0.001), np.pi/4, 0, transition_type="spin_es", axis=1)
# A.plot_optical_transitions(np.arange(0, 0.501, 0.001), 0.99*np.pi/2, 0)

# A.plot_transitions_strengths(np.arange(0,0.501,0.001), 0.99*np.pi/2, 0, transition_type="optical", axis=None)
# A.plot_optical_transitions_strengths_fixed_B(0.1, np.pi/4, 0, axis=None)

# A.plot_superhyperfine(np.arange(0.0005, 0.2, 0.005), 0, 0)
# A.plot_superhyperfine(np.arange(0.0000, 0.0101, 0.0001), 0, 0)

# A.plot_branching_contrast_map_brange(np.linspace(0, 20, 9), np.pi/2, 225/180*np.pi)
# A.plot_max_overlap_rho_map(0.3927, 0*np.pi, grid_size=9, file_in=None)
# A.overlay_atoms()
# A.plot_max_transition_overlap_map(0.975*np.pi/2, 0*np.pi, grid_size=25, file_in="YbTransitionOverlapMap_25x25_87_0.txt")
# A.overlay_atoms()

# A.plot_branching_contrast_angular(1.98678, np.pi, np.arange(-4, 4, 0.002), 0.975*np.pi/2, 0)
# A.plot_branching_contrast_cartesian(A.neighbours[0], np.arange(-1.6, 1.6, 0.001), 0.5*np.pi/2, 0.*np.pi/2)
#10**-10 * np.array((-3.55915, 0.0, -1.57232))
# plt.show()

# max_overlap_neighs = A.neighbours_max_overlap(np.arange(-0.2, 0.2, 0.001), 0.5*np.pi/2, 0, remove_biggest=True)
# A.plot_branching_contrast_cartesian(A.neighbours[0], np.arange(0, 1, 0.001), 0.5*np.pi/2, 0.*np.pi/2)
# print(np.linalg.norm(A.neighbours[0]))
# A.plot_branching_contrast_angular(1.98678, np.pi, np.arange(0, 4, 0.002), 0.975*np.pi/2, 0)
# A.plot_branching_contrast_cartesian(A.neighbours[-1], np.arange(0, 1, 0.001), 0.5*np.pi/2, 0.*np.pi/2)
# plt.plot([i[2][0] for i in max_overlap_neighs])

# A.plot_neighbours_max_overlaprho_Bmap(theta_grid_size=9, phi_grid_size = 9, file_in=None, search_all=False)
# rot = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])

# A.plot_holeburning_spectrum(0.21, 0.99*np.pi/2, 0, (0, 7), 1)
plt.show()