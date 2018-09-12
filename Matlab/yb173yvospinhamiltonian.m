function out = yb173yvospinhamiltonian
% This is a script to keep my final values for the spin hamiltonians so
% that I make sure i'm consistent across my scripts.

    out = spin_hamiltonian_class(1/2,5/2);

    % values of the A tensor are in GHz. 
    % the ground state A and g are from Ranon. 
    out.A_gs = [-0.186, 0,  0;...
                       0, -0.186   0;...
                       0, 0,  1.328]; 


    out.A_es = -[0.959,   0,  0;...
                    0, 0.959,  0;...
                    0,   0,  1.276];

    out.generate_hyperfine_hamiltonian;            

    out.g_gs =   [0.85, 0,  0;...
                      0,  0.85,  0;...
                      0,  0, -6.08]; 

    out.g_es =   [1.6, 0,  0;...
                     0, 1.6,  0;... % 1.6?
                     0,  0, 2.51]; %2.513



end

