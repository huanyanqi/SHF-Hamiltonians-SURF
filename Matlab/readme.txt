I've attached a folder with some example code. The spin_hamiltonian_class file does basically all the work and the rest are helper files. Spin_hamiltonian_test and spin_hamiltonian_branching_ratio_all are some examples of the code in action to calculate energy levels. 

I've also included a matlab app package (YbYVOspinhamiltonianplotter), which is a little GUI that I put together to plot the energy levels + branching ratios for the different isotopes of Yb. This code just covers the zeeman and hyperfine interactions, so it would be something like this expanded to include the superhyperfine interaction. 

To start, it might be useful to think about looking at the superhyperfine between a zero-spin isotope of ytterbium and the surrounding yttrium nuclear spins. i.e. start by ignoring the hyperfine interaction with the ytterbium nuclear spin and just look at the spin-1/2 electron spin coupling with the spin-1/2 yttrium.
