function out = yb171yvospinhamiltonian
% This is a script to keep my final values for the spin hamiltonians so
% that I make sure i'm consistent across my scripts.

    out = spin_hamiltonian_class(1/2,1/2);

    % values of the A tensor are in GHz. 
    % the ground state A and g are from Ranon. 
    out.A_gs = [0.6745, 0,  0;...
                       0, 0.6745,   0;...
                       0, 0,  -4.8205]; 


    out.A_es = [3.39,   0,  0;...
                    0, 3.39,  0;...
                    0,   0,  4.864];

    out.generate_hyperfine_hamiltonian;            

    out.g_gs =   [0.85, 0,  0;...
                      0,  0.85,  0;...
                      0,  0, -6.08]; 

    out.g_es =   [1.6, 0,  0;...
                     0, 1.6,  0;... % 1.6?
                     0,  0, 2.51]; %2.513



end

