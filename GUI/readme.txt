Requirements
================================================================================
- Python (preferably Python 3 and above, have not been tested extensively on 
  Python 2 but it should work)

Instructions
================================================================================
- Run HoleburnGUI.py
- Enter the relevant parameters and press Compute & Plot Spectrum
- Alternatively, if we already have a _peaks.txt, _prelimspectrum.txt or 
  _spectrum.txt file, we can just use Load Theory Spectrum/Peaks to load them and Plot Spectrum.
- Optional: Load an experiment file using Load Experiment, which will be plotted
  together with the theory when Compute & Plot Spectrum or Plot Spectrum is called. 

Explanation of Features
================================================================================
- This GUI has four main features: 
    1. [Compute & Plot Holeburning Spectrum] 
    By inputting the configuration of the system (rare earth ion spins, Zeeman
    and hyperfine tensors, neighbour locations and atom types, types of 
    transitions that we are pumping on, B field, and E field polarisation), it can
    compute the energies and intensities of the holes and antiholes. 

    2. [Plot Holeburning Spectrum]
    A subset of the previous functionality. We can load a previously computed set 
    of holes/antiholes using "Load Theory Spectrum/Peaks" and it will plot the spectrum. 
    We can also load a set of experimental data to overlay onto the plot.

    3. [Plot Optical Transitions & with Strengths]
    Plots the optical transition energies as a function of magnetic field strength.
    The B field direction is given by the specified theta and phi, and the maximum
    magnitude is given by the specified B. "with Strengths" plots the optical 
    transitions but with black shading with darker lines indicating stronger 
    transition strengths. The E field polarisation is given by the E field pol.

    4. [Print Energies]
    Updates the B field using the specified B field and prints in the console the
    list of energy levels.

- System parameters:
    * All g factors are dimensionless. They are used like mu = -mu_B g.S where mu 
    is the magnetic moment of an electron, mu_B is the Bohr magneton, and S is the
    spin. For the neutrons we have mu = mu_N g.I where I is the nuclear spin.
    The negative sign for the electron is because of its negative charge.
    * All A factors are in units of GHz.
    * The range start/end parameters work as follows: suppose range for ground is
    set as 0-8 and range for excited is set 8-16. This follows Python range notation
    where the last index is excluded, so we actually have 0-7 and 8-15. This set
    of settings means that because of inhomogeneous broadening, the holeburning 
    laser will pump between every possible pair of energy levels from these ranges,
    i.e. 0(g) -> 8(e), 9(e),...,15(e), 1(g) -> 8(e),...,15(e), ..., 7(g) -> 8(e),...,15(e)

- Plotting parameters:
    * Each hole/antihole is modelled as a Lorentzian with peak 1 and specified FWHM,
    so each one is not normalised since it has constant height regardless of FWHM!
    * The overall height of all peaks are *divided* by the damping ratio.
    * Convolution is done if the checkbox is ticked and it uses a rectangular
    box located at [-convolution radius, convolution radius] with height 1/6.
    * The x limits can be set but take note that it only recomputes the actual
    spectrum data if we start from a _peaks.txt file! This means that if we used
    a _peaks.txt file and had a more zoomed out setting, and then input a smaller
    x range while using the generated _prelimspectrum.txt or _spectrum.txt file,
    we would simply be stretching the spectrum and might start to quickly see 
    individual points.
    * Warning: When using a _prelimspectrum.txt file, the FWHM field will be
    ignored. Similarly, when using a _spectrum.txt file, all plotting paramters
    except the x limits will be ignored.

- Input/Output files
    * All out spectrum and preliminary spectrum files have the following format:
        # Two columns, first header row is a description of the columns
        # First column is the energy in GHz, second column is the transmission intensity.
    * The input experiment data files have the following format:
        # At least 3 columns, no description header row, delimited by commas 
        # First column is the energy in GHz (assumed to be properly centered),
        second column is the transmission before holeburning and the third
        column is the transmission after holeburning. We plot the quantity
        ln(transmission after holeburning/transmission before holeburning)
        since we know that transmission = e^(-alpha L) so we can get a quantity 
        proportional to alpha by taking the log of the intensity ratio. We are
        predicting something proportional to alpha, the absroption coefficient 
        in the theory since it is proportional to population level among other things.
        The negative is handled by the fact that we make holes positive and 
        antiholes negative.
    * When we do Compute & Plot Spectrum, it will first output a file in the same
    directory as the GUI file that is named with the settings and will end in 
    _peaks.txt. This contains a list of holes/antiholes in the following format:
        # Four columns, first header row is a description of the columns
        # First column is the index of the ground stae that the transition starts
        from, the second column is the index of the excited state for the transition.
        The third colum is the transition energy in GHz and the fourth column is the
        transition strength based on the state overlap amplitude and the population 
        level. It is in arbitrary units.
    * When plotting, the program will read the _peaks.txt file and first use the
    given FWHM and xmin & xmax to create the Lorentzians for each peak and create
    a file ending in _prelimspectrum.txt. This file has the format as specified above.
    * Finally, the program will read in the _prelimspectrum.txt file and use the 
    convolution settings and the damping ratio to modify the spectrum and save the
    final spectrum in a _spectrum.txt file. The reason for splitting the plotting
    into 2 steps is that it is often necessary to try many values of damping ratio
    or convolution width to optimise the fit, but it is time-consuming to repeat
    the first part of Lorentzian construction since we often have a large number
    of peaks. 