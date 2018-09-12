import numpy as np
import copy
import  math
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

# SMALL_SIZE = 24
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 30
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

class SpinSystem:

    mu_b = 9.274*10**-24 # Bohr Magneton (J/T)
    h = 6.626*10**-34   # Planck's constant (J s)
    beta_el = mu_b / (h * 10**9) # Bohr Magneton in GHz/T

    mu_n = 5.050*10**-27 # Nuclear Magneton (J/T)
    beta_n = mu_n / (h * 10**9) # Nuclear Magneton in GHz/T

    def generate_spin_matrices(self, spin):
        """ Generates the x, y, z spin matrices for an arbitrary spin"""

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

        if not REI_NUCLEAR: self.multI = 1

        # Populate the spin matrices 
        self.sx, self.sy, self.sz = self.generate_spin_matrices(self.spinS)
        # Concatenate them together. Dimensions: 3 x multS x multS
        self.S = np.array([self.sx, self.sy, self.sz])
        # Convert to a 3D vector with dimensions 3 x multS^2
        self.Sv3 = self.S.reshape((3, self.multS**2))

        if REI_NUCLEAR:
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

    def set_vector(self, disp_vector):
        self.__init__(self.spinS, self.spinI, self.g_gs, self.g_es, self.A_gs, self.A_es, self.g_n_rei, self.g_n_host, self.spinH, disp_vector)

    def reset_ham(self):
        """ Sets the Hamiltonian to consist of only the field independent hyperfine 
        and superhyperfine parts. """

        self.H_gs = copy.deepcopy(self.HHF_gs + self.H_SHF_gs)
        self.H_es = copy.deepcopy(self.HHF_es + self.H_SHF_es)

    def initialize_HHF(self):
        if REI_NUCLEAR:
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
        else:
            self.HHF_gs = 0
            self.HHF_es = 0

    def initialize_SHF(self):
        # mu = mu_b * g * S
        # Use v3 so that we can take the product with the 3x3 g matrix
        # Then reshape so that we get back our column 3 x (multS x multS) 
        mu_REI_gs = (self.mu_b * self.g_gs @ self.Sv3).reshape((3, self.multS, self.multS))
        mu_REI_es = (self.mu_b * self.g_es @ self.Sv3).reshape((3, self.multS, self.multS))
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

        if REI_NUCLEAR:
            # Do the same for the REI nuclear Zeeman Hamiltonian but there is no 
            # distinction between GS and ES. We assume the nuclear g factor is an 
            # isotropic scalar.
            HZ_n_rei = (self.beta_n * self.g_n_rei * B_vec @ self.Iv3).reshape((self.multI, self.multI))
            # Expand into the electronic spin space
            HZ_n_rei = np.kron(np.identity(self.multS), HZ_n_rei)
            # Expand into the superhyperfine nuclear spin space
            HZ_n_rei = np.kron(HZ_n_rei, np.identity(self.multH))
        else:
            HZ_n_rei = 0

        # Do the same for the nuclear Zeeman Hamiltonian  for the host nuclear spin
        HZ_n_host = (self.beta_n * self.g_n_host * B_vec @ self.Hv3).reshape((self.multH, self.multH))
        # Expand into the REI nuclear spin space
        HZ_n_host = np.kron(np.identity(self.multI), HZ_n_host)
        # Expand into the electronic spin space
        HZ_n_host = np.kron(np.identity(self.multS), HZ_n_host)

        # Reset the Hamiltonian to be only the field-indep HF part and HHF part
        self.reset_ham()
        # Add in the just-computed Zeeman terms
        self.H_gs += HZ_el_gs - HZ_n_rei - HZ_n_host
        self.H_es += HZ_el_es - HZ_n_rei - HZ_n_host

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
        self.E_gs = sorted(np.linalg.eigvals(self.H_gs))
        self.E_es = sorted(np.linalg.eigvals(self.H_es))

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
                gs_transitions.append(self.E_gs[j]-self.E_gs[i])
        for i in range(len(self.E_es)):
            for j in range(i+1, len(self.E_es)):
                es_transitions.append(self.E_es[j]-self.E_es[i])
        return [gs_transitions, es_transitions]

    def optical_transitions(self, B=None, B_theta=None, B_phi=None):
        """ Computes the transitions between the energy levels. Optional to
        provide a B field to update before computing. """

        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        transitions = []

        for i in self.E_gs:
            for j in self.E_es:
                transitions.append(j-i)
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

        return lower, upper, diff

    def compute_branching_contrast(self, B=None, B_theta=None, B_phi=None):
        """ Computes the branching contrast of the system (given its current 
        displacement vector of the host atom) using the formula given in Car. """
        if B is not None and B_theta is not None and B_phi is not None:
            self.update_B(B, B_theta, B_phi)

        # Get the ground state eigvals and eigvecs
        eigval_gs, eigvec_gs = np.linalg.eig(self.H_gs)
        # Sort the columns of the eigvecs in increasing size of the eigvals
        eigvec_gs = eigvec_gs[:, np.argsort(eigval_gs)]
        gs_lower = eigvec_gs[:, 0]
        gs_upper = eigvec_gs[:, 1]

        # Do the same for the excited state
        eigval_es, eigvec_es = np.linalg.eig(self.H_es)
        # Sort the columns of the eigvecs in increasing size of the eigvals
        eigvec_es = eigvec_es[:, np.argsort(eigval_es)]
        es_lower = eigvec_es[:, 0]
        es_upper = eigvec_es[:, 1]

        R = abs(gs_upper.conjugate() @ es_lower.T)**2 / abs(gs_lower.conjugate() @ es_lower.T)**2
        rho = 4 * R / (1 + R)**2

        return abs(gs_upper.conjugate() @ es_lower.T)**2, abs(gs_lower.conjugate() @ es_lower.T)**2, R, rho
        # return rho

    def max_branching_contrast(self, B_range, B_theta, B_phi, remove_biggest):
        """ Looks over a certain range of B fields and try to find the maximum
        value that the branching contrast rho takes for the current displacement
        vector in the self object. Has an option to remove the largest point to 
        remove any spurious points from B=0. """

        # The [3] is to take the rho part of the output from the compute_branching_contrast
        # outoput. [-1] and [-2] take the largest and 2nd largest rho respectively.
        if remove_biggest:
            return sorted([self.compute_branching_contrast(b, B_theta, B_phi)[3] for b in B_range])[-2]
        else:
            return sorted([self.compute_branching_contrast(b, B_theta, B_phi)[3] for b in B_range])[-1]

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
        plt.plot(B_range, np.array([self.optical_transitions(b, B_theta, B_phi) for b in B_range]))
        plt.title("Optical Transitions between energy levels")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")

    def plot_spin_transitions(self, B_range, B_theta, B_phi):
        """ Plot all the spin transition energies as a function of B field. """
        plt.figure()
        plt.suptitle("Spin Transitions within energy level")
        plt.subplot(211)
        plt.plot(B_range, np.array([self.spin_transitions(b, B_theta, B_phi)[1] for b in B_range]))
        plt.title("Excited State Transitions")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")
        plt.subplot(212)
        plt.plot(B_range, np.array([self.spin_transitions(b, B_theta, B_phi)[0] for b in B_range]))
        plt.title("Ground State Transitions")
        plt.xlabel("B field / T")
        plt.ylabel("Transition Energy / GHz")

    def plot_superhyperfine(self, B_range, B_theta, B_phi):
        """ Plot the superhyperfine splitted levels for the lowest energy levels
        (i.e. plot the lowest 2 energy levels after incorporating the SHF
        interaction). In addition, plot the energy gap between these SHF-split
        energy levels.
        """
        fig=plt.figure()
        fig.subplots_adjust(hspace=0.5)
        plt.suptitle("Superhyperfine splittings", y=1.0005)

        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax1.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[:2] for b in B_range])#, c='tab:blue')
        ax1.set_title("Relative Energies of ground\nsuperhyperfine doublet")
        ax1.set_xlabel("B field / T")
        ax1.set_ylabel("Transition Energy / kHz")
        # ax1.set_ylim([-200, 200])

        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax2.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[:2] for b in B_range])#, c='tab:orange')
        ax2.set_title("Relative Energies of excited\nsuperhyperfine doublet")
        ax2.set_xlabel("B field / T")
        ax2.set_ylabel("Transition Energy / kHz")
        # ax2.set_ylim([-200, 200])

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.plot(B_range, [self.superhyperfine_levels('gs', b, B_theta, B_phi)[2] for b in B_range])
        ax3.plot(B_range, [self.superhyperfine_levels('es', b, B_theta, B_phi)[2] for b in B_range])
        ax3.set_title("Energy gap between the superhyperfine doublet levels")
        ax3.set_xlabel("B field / T")
        ax3.set_ylabel("Transition Energy / kHz")
        # ax3.set_ylim((0,300))

    def plot_max_branching_contrast_map(self, B_theta, B_phi, file_in=None):
        """ Plot the theta-phi map of the maximum branching contrast rho (as the B
        field is varied). """ 
        
        if file_in is None:
            # Size of grid in theta-phi space we want to iterate over
            grid_size = 25

            max_branching_contrast_map = np.zeros((grid_size, grid_size))
            phis = np.linspace(0, 2*np.pi, grid_size)
            thetas = np.linspace(0, np.pi, grid_size)

            # Range of B fields to search over to find the max rho
            # This range depends on the length of the vector used below!
            # This range is ok for 10**-10 as the prefactor
            B_range = np.arange(0, 20, 0.1)

            # Iterate over the grid. Theta = x-axis, Phi = y-axis
            for idx_phi, phi in enumerate(phis):
                for idx_theta, theta in enumerate(thetas):
                    vector = 10**-10 * self.unit_vector(theta, phi)
                    self.set_vector(vector)
                    # Find the max branching contrast at that angle with B field at a  
                    # certain angle then store in the array. 
                    # We remove the biggest to avoid any problems with 0 field.
                    max_branching_contrast_map[idx_phi][idx_theta] =  self.max_branching_contrast(B_range, B_theta, B_phi, remove_biggest=True)

            # with open("ErbiumBranchingRatioMap.txt", "w") as f:
                # f.write(max_branching_contrast_map)
            np.savetxt('ErbiumBranchingRatioMap.txt', max_branching_contrast_map)
        else:
            max_branching_contrast_map = np.loadtxt(file_in)
            grid_size = len(max_branching_contrast_map)
            phis = np.linspace(0, 2*np.pi, grid_size)
            thetas = np.linspace(0, np.pi, grid_size)

        # Theta = x-axis, Phi = y-axis
        plt.figure()
        plt.contourf(thetas, phis, max_branching_contrast_map, 100)
        plt.xlabel("Theta")
        plt.ylabel("Phi")
        plt.colorbar()

    def overlay_atoms(self):
        """ Overlay the position of atoms onto the theta-phi map. Takes the 
        Cartesian coordinates of atoms and converts them into angular coordinatess.
        To be used with the plot_max_branching_contrast_map function. """

        # First 15 rows are generated for Y2SiO5 crystal structure 
        neighs = [
        # Generated from VESTA with settings 14
        # (-1.90331, -2.69161, -0.81312), 
        # (-3.26633, 1.13758, 0.0), 
        # (-0.76277, 2.3869, 2.4528), 
        # (2.97802, 1.13758, -1.72032), 
        # (-2.50356, -1.24931, 2.4528), 
        # (-1.36302, 3.82919, -0.81312), 
        # (3.74079, -1.24931, 2.54688), 
        # (-0.76277, 2.3869, -4.2672), 
        # (-2.50356, -1.24931, -4.2672),
        # (4.34104, -2.69161, -0.9072), 
        # (-1.14054, -5.0785, 1.63968), 
        # (1.14055, 5.0785, 1.63968), 
        # (1.83747, -3.94092, -3.36), 
        # (1.83747, -3.94092, 3.36),  
        # (3.74079, -1.24931, -4.17312), 

        # Generated from VESTA with settings 12
        # (a, b, c) = (12.50,  6.72, 10.42)
        # beta = 102.68 / 180 * np.pi 
        # Different from the CIF file!
        (1.90612, -0.81312, -2.69396),
        (3.26883, 0.0, 1.13858),
        (0.7625, 2.4528, 2.38898),
        (-2.98117, -1.72032, 1.13858),
        (2.50633, 2.4528, -1.2504),
        (1.36271, -0.81312, 3.83253),
        (-3.74367, 2.54688, -1.2504),
        (0.7625, -4.2672, 2.38898),
        (2.50633, -4.2672, -1.2504),
        (-4.34388, -0.9072, -2.69396),
        (1.14362, 1.63968, -5.08294),
        (-1.14363, 1.63968, 5.08293),
        (-1.83755, -3.36, -3.94436),
        (-1.83755, 3.36, -3.94436),
        (-3.74367, -4.17312, -1.2504)
        ]
            
        # These are the crystal parameters from Li, Wyon, Moncorge IEEE J. Quantum Electronics 28, 4, Apr 1992
        (a, b, c) = (12.50,  6.72, 10.42)
        beta = 102.68 / 180 * np.pi 
        # Angle between the a,c axes and the D1, D2 axes
        aa = 23.8/180*np.pi
        bb = 78.65/180*np.pi

        # Convert the x,y,z cartesian to u,v,w coordinates in the a,b,c vector basis
        # https://en.wikipedia.org/wiki/Fractional_coordinates
        neighs = [np.array((1/a * coord[0] - np.cos(beta)/(a * np.sin(beta)) * coord[2], 1/b * coord[1], coord[2] / (c* np.sin(beta)))) for coord in neighs]
        # Scale each side's coordinate by the length since the basis vectors have varying lengths 
        # Here we flip the a and c axes for some reason (??) Either we do this or we add pi  to all the phis
        # This doesn't help to solve the coordinate handedness problem since (-a)x(-c)=a x c = -b
        neighs = [coord * np.array((-a,b,-c)) for coord in neighs]
        # Coordinate transformation to convert a-c to D1-D2, and convert b to the z axis
        neighs = [np.array([[np.cos(aa), 0, np.cos(bb)], [-np.sin(aa), 0, np.sin(bb)], [0, 1, 0]]) @ coord for coord in neighs]

        # Last row is the specified atom from the Car paper)
        neighs.append((-1.01, -5.11, 1.64)) 

        # Compute phi from the x,y coords
        neigh_phi = [math.atan2(coord[1], coord[0]) for coord in neighs]
        # Just to make things positive
        neigh_phi = [phi if phi>0 else phi+2*np.pi for phi in neigh_phi]
        
        # # Except for the last Car atom, we add 0.45 offset to each atom
        # neigh_phi = [math.atan2(coord[1], coord[0]) + 0.45 for coord in neighs[:-1]] + [math.atan2(neighs[-1][1], neighs[-1][0])]
        # # Just to make things positive
        # neigh_phi = [phi if phi>0 else phi+2*np.pi for phi in neigh_phi]
        # # Then do a reflection about the phi=pi axis for except the last atom
        # neigh_phi = [2*np.pi-phi for phi in neigh_phi[:-1]] + [neigh_phi[-1]]

        neigh_theta = [math.acos(coord[2]/np.linalg.norm(np.array(coord))) for coord in neighs]

        # All atoms are plotted white except for red for the special Car atom
        plt.scatter(neigh_theta[:-1], neigh_phi[:-1], c='white')
        plt.scatter(neigh_theta[-1], neigh_phi[-1], marker='x', c='red')

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
                    branching_contrast_map[idx_phi][idx_theta] =  self.compute_branching_contrast(B_range[i-1], B_theta, B_phi)[3]

            plt.subplot("33"+str(i))
            plt.contourf(thetas, phis, branching_contrast_map, 100, vmin=0, vmax=1)
            plt.xlabel("Theta")
            plt.ylabel("Phi")
            plt.colorbar()

    def plot_branching_contrast(self, theta, phi, B_range, B_theta, B_phi):
        """ Plots the branching contrast rho as a function of B for a particualr 
        fixed point in theta-phi angular space. """
        vector = 10**-10  * self.unit_vector(theta, phi)
        self.set_vector(vector)
        plt.figure()
        plt.plot(B_range, [self.compute_branching_contrast(b, B_theta, B_phi)[0] for b in B_range], 'ro', label="23")
        plt.plot(B_range, [self.compute_branching_contrast(b, B_theta, B_phi)[1] for b in B_range], 'bo', label="13")
        plt.plot(B_range, [self.compute_branching_contrast(b, B_theta, B_phi)[2] for b in B_range], 'k^', label="R")
        plt.plot(B_range, [self.compute_branching_contrast(b, B_theta, B_phi)[3] for b in B_range], 'gv', label="rho")
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

################################################################################

yb171_A_gs = np.array([[0.6745, 0,  0],
                       [0, 0.6745,   0],
                       [0, 0,  -4.8205]])

yb171_A_es = np.array([[3.39,   0,  0],
                        [0, 3.39,  0],
                        [0,   0,  4.864]])

yb171_g_gs = np.array([[0.85, 0,  0],
                       [0,  0.85,  0],
                       [0,  0, -6.08]])

yb171_g_es = np.array([[1.6, 0,  0],
                       [0, 1.6,  0],
                       [0,  0, 2.51]])

# Taken from the Er g-matrices Site 1 Orientation II from Sun et al PRB 77 (2008) 085124
er_g_gs =   np.array(   [[3.070, -3.124,  3.396],
                         [-3.124,  8.156,  -5.756],
                         [3.396,  -5.756, 5.787]])

er_g_es =   np.array(   [[1.950, -2.212, 3.584],
                         [-2.212, 4.232,  -4.986],
                         [3.584,  -4.986, 7.888]])

REI_NUCLEAR = False

electron_zeeman = 1
hyperfine = 0
nuclear_zeeman_rei = 0
nuclear_zeeman_host = 1
superhyperfine = 1

# Standard vector = (-1.01, -5.11, 1.64)
# r = 5.46, Theta = 1.26577, Phi = 4.51726
A = SpinSystem(
    1/2,                                                            # REI Electronic Spin
    1/2,                                                            # REI Nuclear Spin
    electron_zeeman * er_g_gs,                                   # REI Electronic GS g matrix
    electron_zeeman * er_g_es,                                   # REI ELectronic ES g matrix
    hyperfine * yb171_A_gs,                                         # GS Hyperfine A matrix
    hyperfine * yb171_A_es,                                         # ES Hyperfine A matrix
    nuclear_zeeman_rei * 0.987,                                     # REI Nuclear g coeff
    nuclear_zeeman_host * 0.2753,                                    # Host Nuclear g coeff
    1/2,                                                            # Host Nuclear spin
    superhyperfine * 10**-10 * np.array((-1.01, -5.11, 1.64)) # Displacement vector to host
    )


# A.plot_energies_brange(np.arange(0,0.101,0.001), np.pi/2, 220/180*np.pi)
# A.plot_spin_transitions(np.arange(0,0.101,0.001), 0, 0)
# A.plot_optical_transitions(np
# .arange(0,1.01,0.01), 0, 0)
A.plot_superhyperfine(np.arange(0.001, 0.101, 0.001), np.pi/2, 225/180*np.pi)
# A.plot_branching_contrast_map_brange(np.linspace(0, 20, 9), np.pi/2, 225/180*np.pi)
# A.plot_max_branching_contrast_map(np.pi/2, 225/180*np.pi)
# A.overlay_atoms()
# A.plot_branching_contrast(1.26577, 4.51726, np.arange(-20, 20, 0.01), np.pi/2, 225/180*np.pi)
# A.plot_eigenvector_comps(np.arange(0.001,0.1, 0.001), np.pi/2, 225/180*np.pi)

# A.plot_max_branching_contrast_map(np.pi/2, 225/180*np.pi, 'ErbiumBranchingRatioMap.txt')
# A.overlay_atoms()
plt.show()
