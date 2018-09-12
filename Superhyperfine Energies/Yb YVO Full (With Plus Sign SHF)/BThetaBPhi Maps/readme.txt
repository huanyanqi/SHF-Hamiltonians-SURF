- Since we cannot plot theta-phi for transition amplitude since they are distance dependent and thus it does not make sense to plot them in angular space (with a fixed vector length), we can calculate it for each of the N neighbours. However we have another degree of freedom, ie. direction of B. So we have a grid for B direction, and for each direction we loop over all neighbours,and for each neighbour we sweep over B field strengths to get max transition amplitude. 

- Here all the overlaps are with Sx except for one which is noted in the filename. Also we switch strategy to instead find the maximum amplitude times rho to get high values for both. 

- Also tried searching over all 8x7/2 x 8 = 224 state combinations, although this is 224x more time-consuming. Only managed to do for 9x9 B field direction but it seems there always exists some transition that works for any field direction.

TODO: 
- Take the optimum parameters from the B_theta, B_phi maps and then plot the rhos for that orientation in theta-phi space and overlay the other atoms and see their rhos? Or more simply, just take the known B, B_theta, B_phi and then just calculate the amplitudes/rhos for the other atoms.