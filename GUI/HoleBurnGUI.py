from __future__ import division
import sys
import numpy as np

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
else:
    import tkinter as tk
    from tkinter import filedialog

import matplotlib.pyplot as plt
import SpinClassHoleburningBackend as hb

class TKData:
    def __init__(self):
        self.theory_filename = tk.StringVar()
        self.exp_filename = tk.StringVar()

        self.configChoice = tk.StringVar()
        self.configChoice.set("Custom")
        self.physics_toggle = [tk.IntVar() for i in range(3)]
        self.neighbour_toggle = [tk.IntVar() for i in range(4)]
        self.transition_toggle = [tk.IntVar() for i in range(4)]

        self.widgets = []

        self.A_gs = [[tk.DoubleVar() for j in range(3)] for i in range(3)]
        self.A_es = [[tk.DoubleVar() for j in range(3)] for i in range(3)]
        self.g_gs = [[tk.DoubleVar() for j in range(3)] for i in range(3)]
        self.g_es = [[tk.DoubleVar() for j in range(3)] for i in range(3)]
        self.el_spin = tk.DoubleVar()
        self.nu_spin = tk.DoubleVar()
        self.g_n = tk.DoubleVar()

        self.B = tk.DoubleVar()
        self.B_theta = tk.DoubleVar()
        self.B_phi = tk.DoubleVar()
        self.pol = tk.StringVar()
        self.pol.set("x")

        self.neighbours_list = [(tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.StringVar()) for neigh in range(4)]
        for neigh in range(4): self.neighbours_list[neigh][5].set("X")
        self.transitions_list = [[tk.IntVar() for i in range(4)] for j in range(4)]

        self.FWHM = tk.DoubleVar()
        self.FWHM.set(0.0001)
        self.damp_factor = tk.DoubleVar()
        self.damp_factor.set(0.1)
        self.convolution = tk.IntVar()
        self.conv_width = tk.DoubleVar()
        self.conv_width.set(0.01)
        self.xmin = tk.DoubleVar()
        self.xmin.set(-0.3)
        self.xmax = tk.DoubleVar()
        self.xmax.set(0.3)

    def copy_preset_into_tkvars(self, tkvar_arr, preset):
        """ Copy an array into an array of TKvar variables. """
        # Assume 3x3 matrix
        for i in range(3):
            for j in range(3):
                tkvar_arr[i][j].set(preset[i][j])

    def tkvar_arr_to_arr(self, tkvar_arr):
        """ Create an array from an array of TKVar variables. """
        # Assume 3x3 matrix
        out_arr = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                out_arr[i][j] = tkvar_arr[i][j].get()
        return out_arr

    def copy_preset_neighs_to_tkvars(self, preset):
        """ Convert an array of neighbour tuples into a flattened array of TKvar
        neighbour variables. """
        # There are 4 neighbour rows and each row has 6 fields
        # (x,y,z) in Angstroms, nuclear spin, nuclear g factor, and atom type 
        for neigh in range(4):
            for field in range(6):
                # The first 3 fields are originally in a tuple (x,y,z) and handled differently
                if field in (0, 1, 2):
                    self.neighbours_list[neigh][field].set(preset[neigh][0][field])
                else:
                    self.neighbours_list[neigh][field].set(preset[neigh][field-2])

    def get_default_configs(self, default_config):
        """ Sets the preset parameters from the hb backend module into the TKvars
        of the current instance. """
        if default_config == "Yb-171":
            self.copy_preset_into_tkvars(self.A_gs, hb.yb171_A_gs)
            self.copy_preset_into_tkvars(self.A_es, hb.yb171_A_es)
            self.copy_preset_into_tkvars(self.g_gs, hb.yb171_g_gs)
            self.copy_preset_into_tkvars(self.g_es, hb.yb171_g_es)
            self.el_spin.set(hb.yb171_electron_spin)
            self.nu_spin.set(hb.yb171_nuclear_spin)
            self.g_n.set(hb.yb171_g_nuclear)
            self.copy_preset_neighs_to_tkvars(hb.yb171_neighbours_list)
        else:
            raise NotImplementedError

class Data:
    pass

class Application(tk.Frame):
    def setFilename(self, var):
        """ Function for a button to get a file and store the name into a variable. """
        var.set(filedialog.askopenfilename())

    def plotSpectrum(self):
        """ Plot the theoretical holeburning spectrum and experimental data (optional). """

        # Plot the experimental data if there is one 
        if self.TKdata.exp_filename.get() != "":
            hb.SpinSystem.plot_experimental_spectrum(IN_FILE=self.TKdata.exp_filename.get())

        # Plot the theory file
        # If the ending is _peaks.txt, it means we need to first preprocess to conver the transition lines
        # into an actual spectrum. Here thw FWHM is used. After that, or if we start
        # from a _prelimspectrum.txt file, we will proceed to do the damping + 
        # convolution and save the final result as a _spectrum.txt file and plot.
        # We can also plot directly as a _spectrum.txt file and ignore all plotting configs.
        filename_split = self.TKdata.theory_filename.get().split("_")
        if filename_split[-1] == "peaks.txt":
            hb.SpinSystem.generate_spectrum_from_peaks(IN_FILE=self.TKdata.theory_filename.get(), FWHM=self.TKdata.FWHM.get(), xmin=self.TKdata.xmin.get(), xmax=self.TKdata.xmax.get())
            filename_split[-1] = "prelimspectrum.txt"
            PRELIM_SPEC_FILE = "_".join(filename_split)
            hb.SpinSystem.plot_holeburning_spectrum(IN_FILE=PRELIM_SPEC_FILE, damp_factor=self.TKdata.damp_factor.get(), 
                convolution=self.TKdata.convolution.get(), conv_width=self.TKdata.conv_width.get(), linewidth=2, PLOT_ONLY=False)

        elif filename_split[-1] == "prelimspectrum.txt":
            hb.SpinSystem.plot_holeburning_spectrum(IN_FILE=self.TKdata.theory_filename.get(), damp_factor=self.TKdata.damp_factor.get(), 
                convolution=self.TKdata.convolution.get(), conv_width=self.TKdata.conv_width.get(), linewidth=2, PLOT_ONLY=False)
            plt.xlim([self.TKdata.xmin.get(), self.TKdata.xmax.get()])

        elif filename_split[-1] == "spectrum.txt":
            hb.SpinSystem.plot_holeburning_spectrum(IN_FILE=self.TKdata.theory_filename.get(), damp_factor=1, 
                convolution=False, conv_width=0, linewidth=2, PLOT_ONLY=True)
            plt.xlim([self.TKdata.xmin.get(), self.TKdata.xmax.get()])
        else:
            # If it does not fit, we assume it is the final processed spectrum file
            # but we raise a warning.
            raise ImportWarning("The file does not end with any of the recognised endings. It will be plotted as a spectrum file.")
            hb.SpinSystem.plot_holeburning_spectrum(IN_FILE=self.TKdata.theory_filename.get(), damp_factor=1, 
                convolution=False, conv_width=0, linewidth=2, PLOT_ONLY=True)
            plt.xlim([self.TKdata.xmin.get(), self.TKdata.xmax.get()])

        plt.show()

    def updateFromTKData(self):
        """ Read from the tkinter data types in self.TKdata and convert to the 
        conventional data types to be stored in self.data. Then creates a 
        SpinSystem object and updates its B field so that we can use it for
        all the computations needed. """

        self.data.A_gs = self.TKdata.tkvar_arr_to_arr(self.TKdata.A_gs)
        self.data.A_es = self.TKdata.tkvar_arr_to_arr(self.TKdata.A_es)
        self.data.g_gs = self.TKdata.tkvar_arr_to_arr(self.TKdata.g_gs)
        self.data.g_es = self.TKdata.tkvar_arr_to_arr(self.TKdata.g_es)

        self.data.el_spin = self.TKdata.el_spin.get()
        self.data.nu_spin = self.TKdata.nu_spin.get()
        self.data.g_n = self.TKdata.g_n.get()
        self.data.electron_zeeman, self.data.nuclear_zeeman, self.data.hyperfine = [option.get() for option in self.TKdata.physics_toggle]

        # Get neighbour information into neighbours_list for the checked neighbours
        self.data.neighbours_list = []
        for neigh_idx in range(4):
            if not self.TKdata.neighbour_toggle[neigh_idx].get():
                continue
            self.data.neighbours_list.append(
                ((self.TKdata.neighbours_list[neigh_idx][0].get(),self.TKdata.neighbours_list[neigh_idx][1].get(), self.TKdata.neighbours_list[neigh_idx][2].get()),
                self.TKdata.neighbours_list[neigh_idx][3].get(), self.TKdata.neighbours_list[neigh_idx][4].get(), self.TKdata.neighbours_list[neigh_idx][5].get()))

        # Get transition ranges into the blobs for the checked ranges
        self.data.gs_blobs = []
        self.data.es_blobs = []
        for transition_idx in range(4):
            if not self.TKdata.transition_toggle[transition_idx].get():
                continue
            self.data.gs_blobs.append(range(self.TKdata.transitions_list[transition_idx][0].get(), self.TKdata.transitions_list[transition_idx][1].get()))
            self.data.es_blobs.append(range(self.TKdata.transitions_list[transition_idx][2].get(), self.TKdata.transitions_list[transition_idx][3].get()))

        # Create the spin system object
        self.data.A = hb.SpinSystem(
            self.data.el_spin,                                  # REI Electronic Spin
            self.data.nu_spin,                                  # REI Nuclear Spin
            self.data.electron_zeeman * self.data.g_gs,         # REI Electronic GS g matrix
            self.data.electron_zeeman * self.data.g_es,         # REI ELectronic ES g matrix
            self.data.hyperfine * self.data.A_gs,               # GS Hyperfine A matrix
            self.data.hyperfine * self.data.A_es,               # ES Hyperfine A matrix
            self.data.nuclear_zeeman * self.data.g_n,           # REI Nuclear g coeff
            self.data.neighbours_list                           # List of neighbours
            )

        # Get the B field and polarisation information
        self.data.B = self.TKdata.B.get()
        self.data.B_theta = self.TKdata.B_theta.get()
        self.data.B_phi = self.TKdata.B_phi.get()
        self.data.pol = self.TKdata.pol.get()
        # Update the spin system object
        self.data.A.update_B(self.data.B, self.data.B_theta, self.data.B_phi)

    def computePlotSpectrum(self):
        """ Computes the holeburning spectrum from the input physical parameters
        and then plots the results. """
        self.updateFromTKData()
        self.data.A.compute_holeburning_spectrum(self.data.B, self.data.B_theta, self.data.B_phi, self.data.pol, self.data.gs_blobs, self.data.es_blobs)
        self.TKdata.theory_filename.set("SpectrumData_{}_{}_{}_{}_{}{}_peaks.txt".format(self.data.B,self.data.B_theta, self.data.B_phi, self.data.pol, len(self.data.neighbours_list), self.data.A.neighbours_string))
        self.plotSpectrum()

    def plotOptical(self):
        """ Plots optical transitions from 0 field to the B field magnitude 
        specified. """
        self.updateFromTKData()
        self.data.A.plot_optical_transitions(np.linspace(0, self.data.B, 300), self.data.B_theta, self.data.B_phi)
        plt.show()

    def plotOpticalStrengths(self):
        """ Plots optical transitions from 0 field to the B field magnitude 
        specified with darker colours indicating stronger transitions. The E field
        polarisation is given by the specified axis. """
        self.updateFromTKData()
        axis_dict = {"x": 0, "y": 1, "z": 2}
        self.data.A.plot_transitions_strengths(np.linspace(0, self.data.B, 300), self.data.B_theta, self.data.B_phi, "optical", axis_dict[self.data.pol])
        plt.show()

    def printEnergies(self):
        """ Prints the energy spectrum at the current B field. """
        self.updateFromTKData()
        print("Ground state energies / GHz")
        for i in self.data.A.E_gs: print(i)
        print("\nExcited state energies / GHz")
        for i in self.data.A.E_es: print(i)

    def configChanged(self, *args):
        """ Used as the trace function for the configChoice field that stores the 
        parameters. Upon reaching a config that is preset, we disable all the 
        physical parameter fields. """

        if self.TKdata.configChoice.get() == "Custom":
            for widget in self.TKdata.widgets:
                widget.config(state="normal")

        else:
            for widget in self.TKdata.widgets:
                widget.config(state="disabled")
            self.TKdata.get_default_configs(self.TKdata.configChoice.get())

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.TKdata = TKData()
        self.data = Data()

        self.grid() 
        self.createFirstColumn()
        self.createSecondColumn()
        self.createThirdColumn()
        self.createFinalButtons()
        self.addButtonLogic()

    def createFirstColumn(self):
        tk.Label(self, text="Theory Filename").grid(column=0, row=0, sticky=tk.E)
        self.loadIntensitiesButton = tk.Button(self, text='Load Theory Spectrum/Peaks', command=lambda: self.setFilename(self.TKdata.theory_filename))
        self.loadIntensitiesButton.grid(column=1, row=0, sticky=tk.W)
        self.filenameLabel = tk.Label(self, textvariable=self.TKdata.theory_filename, width=30, wraplength=200, justify=tk.LEFT)
        self.filenameLabel.grid(column=0, columnspan=2, row=1)

        tk.Label(self, text="Experiment Filename").grid(column=0, row=2, sticky=tk.E)
        self.loadIntensitiesButton = tk.Button(self, text='Load Experiment', command=lambda: self.setFilename(self.TKdata.exp_filename))
        self.loadIntensitiesButton.grid(column=1, row=2, sticky=tk.W)
        self.filenameLabel = tk.Label(self, textvariable=self.TKdata.exp_filename, width=30, wraplength=200, justify=tk.LEFT)
        self.filenameLabel.grid(column=0, columnspan=2, row=3)

        tk.Label(self, text="Configuration").grid(column=0, row=4, sticky=tk.E)
        self.configDropdown = tk.OptionMenu(self, self.TKdata.configChoice, 'Yb-171','Custom')
        self.configDropdown.configure(width=10)
        self.configDropdown.grid(column=1, row=4, sticky=tk.W)

        tk.Label(self, text="REI Electron Spin").grid(column=0, row=5, sticky=tk.E)  
        self.elSpinEntry = tk.Entry(self, width=10, textvariable=self.TKdata.el_spin)
        self.TKdata.widgets.append(self.elSpinEntry)
        self.elSpinEntry.grid(column=1, row=5, sticky=tk.W)  

        tk.Label(self, text="REI Nuclear Spin").grid(column=0, row=6, sticky=tk.E)  
        self.nuSpinEntry = tk.Entry(self, width=10, textvariable=self.TKdata.nu_spin)
        self.TKdata.widgets.append(self.nuSpinEntry)
        self.nuSpinEntry.grid(column=1, row=6, sticky=tk.W)  

        tk.Label(self, text="Nuclear g factor").grid(column=0, row=7, sticky=tk.E)  
        self.gREI = tk.Entry(self, width=10, textvariable=self.TKdata.g_n)
        self.TKdata.widgets.append(self.gREI)
        self.gREI.grid(column=1, row=7, sticky=tk.W)

        tk.Label(self, text="Electron Zeeman").grid(column=0, row=8, sticky=tk.E)  
        self.elZeemanCheck = tk.Checkbutton(self, variable=self.TKdata.physics_toggle[0])
        self.elZeemanCheck.toggle()
        self.elZeemanCheck.grid(column=1, row=8, sticky=tk.W)  

        tk.Label(self, text="Nuclear Zeeman", justify=tk.RIGHT).grid(column=0, row=9, sticky=tk.E)  
        self.nuZeemanCheck = tk.Checkbutton(self, variable=self.TKdata.physics_toggle[1])
        self.nuZeemanCheck.toggle()
        self.nuZeemanCheck.grid(column=1, row=9, sticky=tk.W)  

        tk.Label(self, text="Hyperfine Interaction").grid(column=0, row=10, sticky=tk.E)  
        self.hyperfineCheck = tk.Checkbutton(self, variable=self.TKdata.physics_toggle[2])
        self.hyperfineCheck.toggle()
        self.hyperfineCheck.grid(column=1, row=10, sticky=tk.W)  

    def createSecondColumn(self):
        tk.Label(self, text="Zeeman g Tensor for ground state").grid(column=2, row=0)  
        self.gGroundFrame = tk.Frame(self)
        self.gGroundFrame.grid(column=2, row=1)
        self.gGroundArray = [[tk.Entry(self.gGroundFrame, width=10, textvariable=self.TKdata.g_gs[i][j]) for j in range(3)] for i  in range(3)]
        for i in range(3):
            for j in range(3):
                self.gGroundArray[i][j].grid(row=i, column=j) 
                self.TKdata.widgets.append(self.gGroundArray[i][j])
        
        tk.Label(self, text="Zeeman g Tensor for excited state").grid(column=2, row=2)  
        self.gExcitedFrame = tk.Frame(self)
        self.gExcitedFrame.grid(column=2, row=3)
        self.gExcitedArray = [[tk.Entry(self.gExcitedFrame, width=10, textvariable=self.TKdata.g_es[i][j]) for j in range(3)] for i  in range(3)]
        for i in range(3):
            for j in range(3):
                self.gExcitedArray[i][j].grid(row=i, column=j) 
                self.TKdata.widgets.append(self.gExcitedArray[i][j])
    
        tk.Label(self, text="Hyperfine A Tensor for ground state / GHz").grid(column=2, row=4)  
        self.AGroundFrame = tk.Frame(self)
        self.AGroundFrame.grid(column=2, row=5)
        self.AGroundArray = [[tk.Entry(self.AGroundFrame, width=10, textvariable=self.TKdata.A_gs[i][j]) for j in range(3)] for i  in range(3)]
        for i in range(3):
            for j in range(3):
                self.AGroundArray[i][j].grid(row=i, column=j) 
                self.TKdata.widgets.append(self.AGroundArray[i][j])
        
        tk.Label(self, text="Hyperfine A Tensor for excited state / GHz").grid(column=2, row=6)  
        self.AExcitedFrame = tk.Frame(self)
        self.AExcitedFrame.grid(column=2, row=7)
        self.AExcitedArray = [[tk.Entry(self.AExcitedFrame, width=10, textvariable=self.TKdata.A_es[i][j]) for j in range(3)] for i  in range(3)]
        for i in range(3):
            for j in range(3):
                self.AExcitedArray[i][j].grid(row=i, column=j) 
                self.TKdata.widgets.append(self.AExcitedArray[i][j])

    def createThirdColumn(self):
        tk.Label(self, text="Neighbours for Superhyperfine Interaction").grid(column=3, row=0, columnspan=2)
        self.neighbourFrame = tk.Frame(self)
        self.neighbourFrame.grid(column=3, row=1, columnspan=2)
        tk.Label(self.neighbourFrame, text="Enable").grid(column=0, row=0)
        tk.Label(self.neighbourFrame, text="x/Angstrom").grid(column=1, row=0)
        tk.Label(self.neighbourFrame, text="y/Angstrom").grid(column=2, row=0)
        tk.Label(self.neighbourFrame, text="z/Angstrom").grid(column=3, row=0)
        tk.Label(self.neighbourFrame, text="Nuclear Spin").grid(column=4, row=0)
        tk.Label(self.neighbourFrame, text="Nuclear g factor").grid(column=5, row=0)
        tk.Label(self.neighbourFrame, text="Atom Type").grid(column=6, row=0)

        self.neighbourEnableArray = [tk.Checkbutton(self.neighbourFrame, variable=self.TKdata.neighbour_toggle[i]) for i in range(4)]
        self.neighbourArray = [[tk.Entry(self.neighbourFrame, width=10, textvariable=self.TKdata.neighbours_list[i][j]) for j in range(6)] for i  in range(4)]
        for i in range(4):
            self.neighbourEnableArray[i].grid(row=i+1, column=0)
            for j in range(6):
                self.neighbourArray[i][j].grid(row=i+1, column=j+1) 
                self.TKdata.widgets.append(self.neighbourArray[i][j])

        tk.Label(self, text="Sets of transitions to/from the ground/excited that are accessed by the pumping").grid(column=3, row=2, columnspan=2)

        self.transitionFrame = tk.Frame(self)
        self.transitionFrame.grid(column=3, row=3, columnspan=2)
        tk.Label(self.transitionFrame, text="Enable").grid(column=0, row=0)
        tk.Label(self.transitionFrame, text="Range start\nGround").grid(column=1, row=0)
        tk.Label(self.transitionFrame, text="Range end\nGround").grid(column=2, row=0)
        tk.Label(self.transitionFrame, text="Range start\nExcited").grid(column=3, row=0)
        tk.Label(self.transitionFrame, text="Range end\nExcited").grid(column=4, row=0)

        self.transitionEnableArray = [tk.Checkbutton(self.transitionFrame, variable=self.TKdata.transition_toggle[i]) for i in range(4)]
        self.transitionArray = [[tk.Entry(self.transitionFrame, width=10, textvariable=self.TKdata.transitions_list[i][j]) for j in range(4)] for i  in range(4)]
        for i in range(4):
            self.transitionEnableArray[i].grid(row=i+1, column=0)
            for j in range(4):
                self.transitionArray[i][j].grid(row=i+1, column=j+1) 

        self.fieldFrame = tk.Frame(self)
        self.fieldFrame.grid(column=3, row=4, columnspan=2)
        tk.Label(self.fieldFrame, text="B field/T").grid(column=0, row=0)
        tk.Label(self.fieldFrame, text="B_theta").grid(column=1, row=0)
        tk.Label(self.fieldFrame, text="B_phi").grid(column=2, row=0)
        tk.Label(self.fieldFrame, text="E field pol").grid(column=3, row=0)
        self.BEntry = tk.Entry(self.fieldFrame, width=10, textvariable=self.TKdata.B)
        self.BEntry.grid(column=0, row=1)
        self.BThetaEntry = tk.Entry(self.fieldFrame, width=10, textvariable=self.TKdata.B_theta)
        self.BThetaEntry.grid(column=1, row=1)
        self.BPhiEntry = tk.Entry(self.fieldFrame, width=10, textvariable=self.TKdata.B_phi)
        self.BPhiEntry.grid(column=2, row=1)
        self.EPolDropdown = tk.OptionMenu(self.fieldFrame, self.TKdata.pol, "x", "y", "z")
        self.EPolDropdown.configure(width=8)
        self.EPolDropdown.grid(column=3, row=1)

        tk.Label(self, text="FWHM / GHz").grid(column=3, row=5, sticky=tk.E)  
        self.FWHMEntry = tk.Entry(self, width=10, textvariable=self.TKdata.FWHM)
        self.FWHMEntry.grid(column=4, row=5, sticky=tk.W)  

        tk.Label(self, text="Damping Ratio").grid(column=3, row=6, sticky=tk.E)  
        self.dampingEntry = tk.Entry(self, width=10, textvariable=self.TKdata.damp_factor)
        self.dampingEntry.grid(column=4, row=6, sticky=tk.W)  

        tk.Label(self, text="Convolution?").grid(column=3, row=7, sticky=tk.E)  
        self.convolutionCheck = tk.Checkbutton(self, variable=self.TKdata.convolution)
        self.convolutionCheck.grid(column=4, row=7, sticky=tk.W)  

        tk.Label(self, text="Convolution Radius / GHz").grid(column=3, row=8, sticky=tk.E)  
        self.convolutionWidthEntry = tk.Entry(self, width=10, textvariable=self.TKdata.conv_width)
        self.convolutionWidthEntry.grid(column=4, row=8, sticky=tk.W)  

        tk.Label(self, text="Lower x limit / GHz").grid(column=3, row=9, sticky=tk.E)
        self.xminEntry = tk.Entry(self, width=10, textvariable=self.TKdata.xmin)
        self.xminEntry.grid(column=4, row=9, sticky=tk.W) 

        tk.Label(self, text="Upper x limit / GHz").grid(column=3, row=10, sticky=tk.E)
        self.xmaxEntry = tk.Entry(self, width=10, textvariable=self.TKdata.xmax)
        self.xmaxEntry.grid(column=4, row=10, sticky=tk.W) 

    def createFinalButtons(self):
        self.runButton = tk.Button(self, text="Compute & Plot\nHoleburning Spectrum", command=self.computePlotSpectrum)
        self.runButton.grid(column=0, row=11)

        self.plotButton = tk.Button(self, text="Plot Holeburning Spectrum", command=self.plotSpectrum)
        self.plotButton.grid(column=1, row=11)

        self.plotButton = tk.Button(self, text="Plot Optical Transitions", command=self.plotOptical)
        self.plotButton.grid(column=2, row=11)

        self.plotButton = tk.Button(self, text="Plot Optical Transitions\nwith strengths", command=self.plotOpticalStrengths)
        self.plotButton.grid(column=3, row=11)

        self.plotButton = tk.Button(self, text="Print Energies", command=self.printEnergies)
        self.plotButton.grid(column=4, row=11)

    def addButtonLogic(self):
        """ Trigger the configChanged function once configChoice is modified. This
        lets us check if we need to update all the physical parmaeters to the 
        preset values and then lock the fields. """ 
        self.TKdata.configChoice.trace('w', self.configChanged)

app = Application() 
app.master.title("Spectral Holeburning") 
app.mainloop()
