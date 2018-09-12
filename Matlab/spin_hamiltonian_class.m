classdef spin_hamiltonian_class < handle
    %spin_hamiltonian_class was created to make calculations involving spin
    %hamiltonians simpler! basically all the calculations done involving
    %the spin hamiltonian are packaged into this. It's definitely not
    %complete/can't handle all cases (especially systems with nuclear spin >1/2). 
    %It's mainly been used to look at the Yb171 system. 
    
    properties % various properties of the system that we want access to. 
        % also a place to store constances. 
        spinS;
        spinI;
        
        % should also include Q term at some point, but that isn't relevant
        % for the 
        A_gs
        A_es
        g_gs
        g_es
        
        units = 1*10^9; % frequency multiplier. 10^9 is GHz 
        mub = 9.274*10^-24; %bohr magneton. (Joules/Tesla)
        h = 6.626*10^-34; % planck's constant (Joules*s second)
        beta
        %beta = obj.mub/(h*obj.units); % to put energies in GHz.
        
        % will have to set it up so you can step over angle as well.
        B_vec  % units in Tesla!
        B_theta 
        B_phi
        B_offset = [0 0 0];
        
        
        energies_gs
        energies_es
        energies_gs_test
        
        %energies_gs_spin0
        %energies_es_spin0
        
        R
        
        states_gs;
        states_es;
        
        transitions
        transitions_spin_gs
        transitions_spin_es
        transition_strengths
        transition_strengths_perpx
        transition_strengths_perpy
        transition_strengths_par
        
        transition_strengths_ll
        
        S   % set of spin matrices as vector [Sx Sy Sz]
        Sv3 % vectorise by reshaping S into 3 columns
        I
        bigS
        bigI
        bigSv3
        
        Hhf_gs;
        Hhf_es;
        
        delx
        delz
        
        gradx
        gradz
        gradabs
        
        transx
        transz
    end
    
    methods
        function obj = spin_hamiltonian_class(spinS,spinI)
            % this function creates a spin hamiltonian object for a system
            % with electron spin spinS and nuclear spin spinI
            % It generates the appropriate spin matrices (i.e. pauli
            % matrices) for the given spin system. 
           
            obj.spinS = spinS;
            obj.spinI = spinI;
            
            [sx, sy, sz] = generatespinoperator(spinS);
            obj.S = [sx, sy, sz];
            obj.Sv3 = reshape(obj.S,[],3).'; % vectorized form
            
            % creating a S operator for the combined hilbert space.
            obj.bigS = kron(obj.S,eye(2*obj.spinI + 1));
            obj.bigSv3 = reshape(obj.bigS,[],3).'; % vectorized form.
            
            [Ix, Iy, Iz] = generatespinoperator(spinI);
            obj.I = [Ix, Iy, Iz];
            % creating an I operator for the combined hilbert space.
            obj.bigI = [kron(eye(2*obj.spinS + 1),Ix),kron(eye(2*obj.spinS + 1),Iy),kron(eye(2*obj.spinS + 1),Iz)];
           
            obj.beta = obj.mub/(obj.h*obj.units);
            
        end
        
        function generate_hyperfine_hamiltonian(obj)
            % generate the hyperfine interaction hamiltonian for the ground
            % and excited states.
            
            
            if ~isempty(obj.A_gs)
                Hhf_temp = obj.A_gs*obj.bigSv3;

                Hhf_temp = reshape(Hhf_temp.',(2*obj.spinS + 1)*(2*obj.spinI+1),[]);
                obj.Hhf_gs = obj.bigI*Hhf_temp';
                %obj.Hhf_gs = obj.bigI*Hhf_temp.';
            end
            
            if ~isempty(obj.A_es)
                Hhfes_temp = obj.A_es*obj.bigSv3;

                Hhfes_temp = reshape(Hhfes_temp.',(2*obj.spinS + 1)*(2*obj.spinI+1),[]);
                obj.Hhf_es = obj.bigI*Hhfes_temp';
            end
        end
        
        function step_magnetic_field_amplitude(obj,B_vec,theta,phi)
            % this function steps over magnetic field amplitude and calculates the 
            % eigenvals of the full spin hamiltonian.
           
            obj.B_vec = B_vec;
            obj.B_theta = theta;
            obj.B_phi = phi;
            
            obj.energies_gs = zeros(length(B_vec),2*(2*obj.spinI+1));
            obj.energies_es = zeros(length(B_vec),2*(2*obj.spinI+1));
            
            %obj.energies_gs_spin0 = zeros(length(B_vec),2);
            %obj.energies_es_spin0 = zeros(length(B_vec),2);
            
            beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            %beta = 1;
            % should also check that A exists?
            % first for the ground state.
            if ~isempty(obj.g_gs)
                % basically just loops over the magnetic field. Creates the
                % zeeman hamiltonian at that field. Adds in the hyperfine
                % hamiltonian to get the full spin hamiltonian. Then just finds the 
                % energies (eigenvals) from this hamiltonian
                for it = 1:length(B_vec)
                    %theta = stepvec(it);
                    B0 = B_vec(it);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;
                    %B = B0*[1 0 0];
                    HZ1 = beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);

                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    %assignin('base','H_gs',H_gs)
                    obj.energies_gs(it,:) = eig(H_gs);
                    %obj.energies_gs_spin0(it,:) = eig(HZ1);
                    
                end
            end
            % then doing the same for the excited state.
            if ~isempty(obj.g_es)
                for it = 1:length(B_vec)
                        %theta = stepvec(it);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]+obj.B_offset;


                        HY1 = beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end
                        obj.energies_es(it,:) = eig(H_es);
                       % obj.energies_es_spin0(it,:) = eig(H_es);
                end
            end
            
            % calls the find_transitions functions to calculate the
            % transitions given the energies.
            obj.find_transitions;
        end
        
        function find_transitions_spin(obj)
            % find the spin transitions in the ground and excited state.
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1); 
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transitions_spin_gs = zeros(length(obj.energies_gs),nchoosek(n_gs,2));
            it = 0; 
            for gs1 = 1:(n_gs-1)
                for gs2 = (gs1+1):n_gs;
                    it = it+1;
                    %disp(it)
                    obj.transitions_spin_gs(:,it)  = -obj.energies_gs(:,gs1) + obj.energies_gs(:,gs2);
                end
            end
            
            obj.transitions_spin_es = zeros(length(obj.energies_es),nchoosek(n_es,2));
            it = 0; 
            for es1 = 1:(n_es-1)
                for es2 = (es1+1):n_es;
                    it = it+1;
                    %disp(it)
                    obj.transitions_spin_es(:,it)  = -obj.energies_es(:,es1) + obj.energies_es(:,es2);
                end
            end
            
        end
        
        function find_transitions(obj)
            % finds the optical transitions.
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1); % modify this... I guess it should be 2*(2 J + 1)?
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transitions = zeros(length(obj.energies_gs(:,1)),n_es*n_gs);

            for exclevel = 1:n_es
                for gndlevel = 1:n_gs
                    obj.transitions(:,n_gs*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                end
            end
        end  
        
        
        function [fig_h,axes_gs, axes_es] = plot_energies_all(obj,fig_h,axes_gs,axes_es)
            if isempty(fig_h)
                fig_h = figure;
            end
            if isempty(axes_gs)
                fig_h;
                axes_gs = subplot(2,1,2);
            end
            if isempty(axes_es)
                fig_h;
                axes_es = subplot(2,1,1);
            end
            
            plot(axes_gs, obj.B_vec, obj.energies_gs )
            title(axes_gs,'ground state energies')
            xlabel(axes_gs,'B (T)')
            ylabel(axes_gs,'GHz')

            plot(axes_es, obj.B_vec, obj.energies_es )
            title(axes_es,'excited state energies')
            xlabel(axes_es,'B (T)')
            ylabel(axes_es,'GHz')
           
        end
        
        function [fig_h,axes_gs, axes_es] = plot_splittings_all(obj,fig_h,axes_gs,axes_es)
            if isempty(fig_h)
                fig_h = figure;
            end
            if isempty(axes_gs)
                fig_h;
                axes_gs = subplot(2,1,2);
            end
            if isempty(axes_es)
                fig_h;
                axes_es = subplot(2,1,1);
            end
            
            plot(axes_gs, obj.B_vec, obj.energies_gs(:,2:end)-obj.energies_gs(:,1:end-1))
            title(axes_gs,'ground state splittings')
            xlabel(axes_gs,'B (T)')
            ylabel(axes_gs,'GHz')

            plot(axes_es, obj.B_vec, obj.energies_es(:,2:end)-obj.energies_es(:,1:end-1))
            title(axes_es,'excited state splittings')
            xlabel(axes_es,'B (T)')
            ylabel(axes_es,'GHz')
           
        end                                
        
        function [fig_h,axes_h] = plot_transitions_all(obj,fig_h,axes_h)
            % set this up for plotting shared ground states!
            if isempty(fig_h)
                fig_h = figure;
            end
            if isempty(axes_h)
                axes_h = axes;
            end
            
            % change this so it isn't B_vec... i.e. use a more general
            % step_vec
            plot(axes_h, obj.B_vec, obj.transitions,'color',[0 0.4470 0.7410])
            title(axes_h,'All transitions')
            xlabel(axes_h,'B (T)')
            ylabel(axes_h,'GHz')     
        end
        
        function find_branching_ratio_theta(obj,B0,theta_vec,phi)
            obj.B_vec = B0;
            obj.B_theta = theta_vec;
            obj.B_phi = phi;
            
            obj.energies_gs = zeros(length(theta_vec),2*(2*obj.spinI+1));
            obj.energies_es = zeros(length(theta_vec),2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),length(theta_vec));
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),length(theta_vec));
            
            beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            
            
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:length(theta_vec)
                    %theta = stepvec(it);
                    theta = theta_vec(it);

                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                    HZ1 = beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    assignin('base','H_gs',H_gs)
                    %obj.energies_gs(it,:) = eig(H_gs);
                    [V, D] = eig(H_gs,'vector');
%                     obj.energies_gs(it,:) = diag(D);
                    obj.energies_gs(it,:)  = D;
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:length(theta_vec)
                        %theta = stepvec(it);
                        theta = theta_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+ obj.B_offset;


                        HY1 = beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
%             obj.R = zeros(length(theta_vec),1);
%             for it = 1:length(theta_vec)
%                 prod14 = dot(obj.states_gs(:,1,it),obj.states_es(:,3,it));
%                 prod13 = dot(obj.states_gs(:,1,it),obj.states_es(:,1,it));
%                 obj.R(it) = (abs(prod14)^2)/(abs(prod13)^2);
%                 
%             end
            
            obj.find_transitions;
        end
        
        function find_branching_ratio_amplitude(obj,B_vec,theta,phi)
            % one of a few different functions to plot the branching
            % ratios. The best one/most up to date one to use is probably
            % the find_transition_strengths_amplitude function.
            obj.B_vec = B_vec;
            obj.B_theta = theta;
            obj.B_phi = phi;
            step_vec = B_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            
            
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    B0 = B_vec(it);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;

                    HZ1 = beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    
                    assignin('base','H_gs',H_gs)
                    
                    %obj.energies_gs(it,:) = eig(H_gs);
                    [V, D] = eig(H_gs);
                    obj.energies_gs(it,:) = diag(D);
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        %theta = theta(it);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;


                        HY1 = beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
            obj.R = zeros(nsteps,1);
            for it = 1:nsteps
                prod14 = dot(obj.states_gs(:,1,it),obj.states_es(:,3,it));
                prod13 = dot(obj.states_gs(:,1,it),obj.states_es(:,1,it));
                obj.R(it) = (abs(prod14)^2)/(abs(prod13)^2);
                
            end
            
            obj.find_transitions;
        end
            

        function find_branching_ratio_relative(obj,B_vec,theta,phi)
            obj.B_vec = B_vec;
            obj.B_theta = theta;
            obj.B_phi = phi;
            step_vec = B_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    B0 = B_vec(it);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;

                    HZ1 = beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    %assignin('base','H_gs',H_gs)
                    %obj.energies_gs(it,:) = eig(H_gs);
                    %eig(H_es)
                    [V, D] = eig(H_gs);
                    %assignin('base','d',D)
                    obj.energies_gs(it,:) = diag(D);
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        %theta = theta(it);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                        HY1 = beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transition_strengths = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perp = zeros(nsteps,n_es*n_gs);
            sx = 2*obj.bigS(:,1:2*(2*obj.spinS + 1));
            ix = 2*obj.bigI(:,1:2*(2*obj.spinI + 1));
            obj.find_transitions;
            for it = 1:nsteps
                for exclevel = 1:n_es
                    for gndlevel = 1:n_gs;
                        % are these properly assigned?
                        % states go low
                        obj.transition_strengths(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),sx*obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths(it,4*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                    end
                    
                end
            end
        end
        
        function find_transition_strengths_amplitude(obj,B_vec,theta_vec,phi)
            % will find the transition strengths (for both polarizations)
            % as you change the field amplitude
            
            obj.B_vec = B_vec;
            obj.B_theta = theta_vec;
            obj.B_phi = phi;
            step_vec = B_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            obj.beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    %B0 = B_vec(it);
                    B0 = B_vec(it); % in case I accidentally have a b vector. 
                    theta = theta_vec(1);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;
                    assignin('base','B',B)
                    HZ1 = obj.beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    %assignin('base','Hz1',HZ1)
                    %obj.energies_gs(it,:) = eig(H_gs);
                    %eig(H_es)
                    [V, D] = eig(H_gs);
                    %assignin('base','d',D)
                    %% explicitly sort them...
                    [d, ind] = sort(diag(D));
                    D = D(ind,ind);
                    V = V(:,ind);
                    
                    obj.energies_gs(it,:) = diag(D);
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        theta = theta_vec(1);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                        HY1 = obj.beta*B*obj.g_es*obj.Sv3;
                        
                        HY1 = reshape(HY1,2,2);
                        %assignin('base','HY1',HY1)
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        %% explicitly sort them...
                        [d, ind] = sort(diag(D));
                        D = D(ind,ind);
                        V = V(:,ind);
                        
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            %assignin('base','H_es',H_es)
            %assignin('base','H_gs',H_gs)
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transition_strengths_par = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpx = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpy = zeros(nsteps,n_es*n_gs);
            
            degen = n_es;
            
            sx = 2*obj.bigS(:,1:degen);
            sy = 2*obj.bigS(:,(1:degen)+degen);
            sz = 2*obj.bigS(:,(1:degen)+2*degen);
            
            disp(sx);
            disp(sy)
            disp(sz)

            obj.find_transitions;
            %assignin('base','sz',sz)
            %assignin('base','sx',sx)
            % calculate the transition strengths for the two polarizations.
            % 
            for it = 1:nsteps
                for exclevel = 1:n_es
                    for gndlevel = 1:n_gs;
                        % are these properly assigned?
                        % states go low
                        %obj.transition_strengths_par(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        obj.transition_strengths_par(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sz*obj.states_gs(:,gndlevel,it)).^2;
                        obj.transition_strengths_perpx(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sx*obj.states_gs(:,gndlevel,it)).^2;
                        obj.transition_strengths_perpy(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sy*obj.states_gs(:,gndlevel,it)).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths(it,4*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                    end
                    
                end
            end
        end
        
        
        function find_transition_strengths_amplitude_v2(obj,B_vec,theta_vec,phi)
            % will find the transition strengths (for both polarizations)
            % as you change the field amplitude
            
            obj.B_vec = B_vec;
            obj.B_theta = theta_vec;
            obj.B_phi = phi;
            step_vec = B_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            H_gs_mat = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            H_es_mat = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            obj.beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    %B0 = B_vec(it);
                    B0 = B_vec(it); % in case I accidentally have a b vector. 
                    theta = theta_vec(1);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;

                    HZ1 = obj.beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    
                    H_gs_mat(:,:,it) = H_gs;
                    %assignin('base','H_gs',H_gs)
                    %obj.energies_gs(it,:) = eig(H_gs);
                    %eig(H_es)
                    %[V, D] = eig(H_gs);
                    %assignin('base','d',D)   
                    
                end
            end
            
            % use eigenshuffle to calculate eigenvals/vectors and sort
            % appropriately
            [Vseq, Dseq] = eigenshuffle(H_gs_mat);
            assignin('base','Vseq',Vseq);
            assignin('base','Dseq',Dseq);
            obj.energies_gs = rot90(Dseq,-1);
            obj.states_gs = fliplr(Vseq); %    
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        theta = theta_vec(1);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                        HY1 = obj.beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               
                        H_es_mat(:,:,it) = H_es;
                        %[V,D] = eig(H_es);
                        
                        %obj.energies_es(it,:) = diag(D);
                        %obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
            [Vseq, Dseq] = eigenshuffle(H_es_mat);
            obj.energies_es = rot90(Dseq,-1);
            obj.states_es = fliplr(Vseq); %   
            
            assignin('base','H_es',H_es)
            assignin('base','H_gs',H_gs)
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transition_strengths_par = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpx = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpy = zeros(nsteps,n_es*n_gs);
            sx = 2*obj.bigS(:,1:2*(2*obj.spinS + 1));
            sy = 2*obj.bigS(:,5:8);
            sz = 2*obj.bigS(:,9:12);
            ix = 2*obj.bigI(:,1:2*(2*obj.spinI + 1));
            obj.find_transitions;
            
            % calculate the transition strengths for the two polarizations.
            % 
            for it = 1:nsteps
                for exclevel = 1:n_es
                    for gndlevel = 1:n_gs;
                        % are these properly assigned?
                        % states go low
                        obj.transition_strengths_par(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(conj(obj.states_es(:,exclevel,it)),sy*obj.states_gs(:,gndlevel,it))).^2;
                        obj.transition_strengths_perpx(it,4*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sx*obj.states_gs(:,gndlevel,it)).^2;
                        obj.transition_strengths_perpy(it,4*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sy*obj.states_gs(:,gndlevel,it)).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths(it,4*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                    end
                    
                end
            end
        end
        
        function find_transition_strengths_fieldangle(obj,B_vec,theta_vec,phi)
            % will find the transition strengths (for both polarizations)
            % as you change the field angle. if B_vec is a vector it will
            % only take the 1st value. probably a good way to handle all
            % this, but 
            
            obj.B_vec = B_vec;
            obj.B_theta = theta_vec;
            obj.B_phi = phi;
            step_vec = theta_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            obj.beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    %B0 = B_vec(it);
                    B0 = B_vec(1); % in case I accidentally have a b vector. 
                    theta = theta_vec(it);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;

                    HZ1 = obj.beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    %assignin('base','H_gs',H_gs)
                    %obj.energies_gs(it,:) = eig(H_gs);
                    %eig(H_es)
                    [V, D] = eig(H_gs);
                    %assignin('base','d',D)
                    obj.energies_gs(it,:) = diag(D);
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        theta = theta_vec(it);
                        B0 = B_vec(1);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                        HY1 = obj.beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            obj.transition_strengths_par = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpx = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_perpy = zeros(nsteps,n_es*n_gs);
            
            
            degen = n_es;
            
            sx = 2*obj.bigS(:,1:degen);
            sy = 2*obj.bigS(:,(1:degen)+degen);
            sz = 2*obj.bigS(:,(1:degen)+2*degen);
            % calculate the transition strengths for the two polarizations.
            % 
            obj.find_transitions;
            for it = 1:nsteps
                for exclevel = 1:n_es
                    for gndlevel = 1:n_gs;
                        % are these properly assigned?
                        % states go low
                        obj.transition_strengths_par(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sz*obj.states_gs(:,gndlevel,it)).^2;
                        obj.transition_strengths_perpx(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sx*obj.states_gs(:,gndlevel,it)).^2;
                        obj.transition_strengths_perpy(it,n_gs*(exclevel-1) + gndlevel) = abs(obj.states_es(:,exclevel,it)'*sy*obj.states_gs(:,gndlevel,it)).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths(it,4*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                    end
                    
                end
            end
        end
        
        function branching_ratios_par = find_branching_ratio_fieldangle(obj,B_vec,theta_vec,phi,exc_vec, gnd_vec)
            % use the function I created above to find all the transition
            % strengths for those different field angles. 
            find_transition_strengths_fieldangle(obj,B_vec,theta_vec,phi)
            
            % then you need to use these transitions to find the branching
            % ratios. how about you define your excited states/ground states of interest
            % (for yb:yvo it'll be 1:4)
            % exc_vec will be excited states of interest. This will be
            % easier if you just pick one. 
            
            n_es = length(exc_vec); 
            n_gs = length(gnd_vec);
            numtrans = n_es*n_gs; % number transitions to find branching ratios for
            nsteps = size(obj.transition_strengths_par,1); 
            
            branching_ratios_par = zeros(nsteps,numtrans);
            
            for it = 1:nsteps
                for ex_it = 1:n_es
                    for gnd_it = 1:n_gs;
                        exclevel = exc_vec(ex_it);
                        gndlevel = gnd_vec(gnd_it);
                        % are these properly assigned?
                        % states go low
                        branching_ratios_par(it,n_gs*(ex_it-1) + gnd_it) = obj.transition_strengths_par(it,n_gs*(exclevel-1) + gndlevel);
              
                    end 
                end
                % then to get relative ratio, divide by the sum
                branching_ratios_par(it,:) = branching_ratios_par(it,:)./sum(branching_ratios_par(it,:));
            end

            
        end
        
        function find_branching_ratio_relative_subset(obj,B_vec,theta,phi)
            %% want to find the branching ratios just a given set of interest. in this case, the like to like. 
            %so looking at lowest energies in the ground state and excited
            %state.
            % should set this up so I can either ramp the field or the
            % angle without having to change anything.
            obj.B_vec = B_vec;
            obj.B_theta = theta;
            obj.B_phi = phi;
            step_vec = B_vec;
            nsteps = length(step_vec);
            
            obj.energies_gs = zeros(nsteps,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nsteps,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nsteps);
            
            beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            % should also check that A exists?
            if ~isempty(obj.g_gs)
                for it = 1:nsteps
                    %theta = stepvec(it);
                    B0 = B_vec(it);
                    B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;

                    HZ1 = beta*B*obj.g_gs*obj.Sv3;
                    HZ1 = reshape(HZ1,2,2);
                    if isempty(obj.Hhf_gs)
                        H_gs =  kron(HZ1,eye(2*obj.spinI+1));
                    else
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                    end
                    
                    %obj.energies_gs(it,:) = eig(H_gs);
                    [V, D] = eig(H_gs);
                    assignin('base','d',D)
                    obj.energies_gs(it,:) = diag(D);
                    obj.states_gs(:,:,it) = V; % columns are the eigenvectors.          
                    
                end
            end
            if ~isempty(obj.g_es)
                for it = 1:nsteps
                        %theta = stepvec(it);
                        %theta = theta(it);
                        B0 = B_vec(it);
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;

                        HY1 = beta*B*obj.g_es*obj.Sv3;
                        HY1 = reshape(HY1,2,2);
                        %HZ1_big = kron(HZ1,ones(2*spinI+1));
                        if isempty(obj.Hhf_es)
                            H_es =  kron(HY1,eye(2*obj.spinI+1));
                        else
                            H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
                        end               

                        [V,D] = eig(H_es);
                        obj.energies_es(it,:) = diag(D);
                        obj.states_es(:,:,it) = V; % columns are the eigenvectors.
                end
            end
            
            n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
            
            %obj.transition_strengths = zeros(nsteps,n_es*n_gs);
            obj.transition_strengths_ll = zeros(nsteps,2);
            sx = 2*obj.bigS(:,1:2*(2*obj.spinS + 1));
            ix = 2*obj.bigI(:,1:2*(2*obj.spinI + 1));
            obj.find_transitions;
            for it = 1:nsteps
                for exclevel = 1
                    for gndlevel = 1:2;
                        % are these properly assigned?
                        % states go low
                        obj.transition_strengths_ll(it, gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths_perp(it,4*(exclevel-1) + gndlevel) = abs(dot(obj.states_es(:,exclevel,it),obj.states_gs(:,gndlevel,it))).^2;
                        %obj.transition_strengths(it,4*(exclevel-1) + gndlevel)  = -obj.energies_gs(:,gndlevel) + obj.energies_es(:,exclevel);
                    end
                    
                end
            end
        end
               
        
        function find_zeeman_gradient(obj,Bx, Bz)
            % The goal here is to calculate the zeeman gradient at each
            % magnetic field. Essentially want to create a grid of magnetic
            % fields, which you'll input into B_x and B_z. I'm assuming a
            % uniaxiaal crystal here, so I'll apologize to my future self
            % for not making it more general!
            % I'm only thinking about the ground state spin transitions here, but you
            % could do a similar thing for the optical...
            % B_mat = [Bx By];
            nx = length(Bx);
            nz = length(Bz);
            
            obj.energies_gs = zeros(nx,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nz,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nx,nz);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nz);
            
            obj.beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            
            % Here I'll put the change in a given field level for perturbation
            obj.delx = zeros(2*(2*obj.spinI+1),nx,nz);
            obj.delz = zeros(2*(2*obj.spinI+1),nx,nz);
            
            nstates = 2*(2*obj.spinI+1);
            
            % my perturbations (i.e. what I need for the perturbation
            % theory) are 
           ptbx = obj.beta*[1,0,0]*obj.g_gs*obj.Sv3;
           ptbx = reshape(ptbx,2,2);
           ptbx = kron(ptbx,eye(2*obj.spinI+1));
          % disp
           
           ptbz = obj.beta*[0,0,1]*obj.g_gs*obj.Sv3;
           ptbz = reshape(ptbz,2,2);
           ptbz = kron(ptbz,eye(2*obj.spinI+1));
           
           obj.transx = zeros(nstates,nstates,nx, nz);
           obj.transz = zeros(nstates,nstates,nx, nz);
            
            % the basic idea here is that I want to step over magnetic
            % fields and at each magnetic field calculate the gradient. 
            for itz = 1:nz
                for itx = 1:nx
                        %theta = stepvec(it);
                        %B0 = B_vec(it);
                        %B0 = B_vec(1); % in case I accidentally have a b vector. 
                        %theta = theta_vec(it);
                        %B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)] + obj.B_offset;
                        B = [Bx(itx), 0, Bz(itz)];
                        
                        HZ1 = obj.beta*B*obj.g_gs*obj.Sv3;
                        HZ1 = reshape(HZ1,2,2);
                        
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                        % calculate the eigenval and vectors at the given
                        % field
                        [V, ~] = eig(H_gs);
                        % Columns of V are the eigenvectors;
                        obj.states_gs(:,:,itx,itz) = V;
                        
                       % then to calculate the gradient, I first need the del for each transition
        
                       for state = 1:nstates;
                           obj.delx(state,itx,itz) = dot(V(:,state),ptbx*V(:,state));
                           obj.delz(state,itx,itz) = dot(V(:,state),ptbz*V(:,state));
                       end
                       
                       % calculate the transition probabilities
                       for s1 = 1:nstates;
                            for s2 = 1:nstates; 
                                obj.transx(s1,s2,itx,itz) = dot(V(:,s1),ptbx*V(:,s2));
                                %obj.transz(s1,s2,itx,itx) = dot(V(:,s1),ptbz*V(:,s2));
                                obj.transz(s1,s2,itx,itz) = dot(V(:,s1),ptbz*V(:,s2));
                            end
                       end
                       
                end
            end
            
            % I then want to use this to calculate the gradient matrices.
            % i.e. the difference in the dels for a given transition. 
            
            obj.gradx = zeros(nstates,nstates,nx, nz);
            obj.gradz = zeros(nstates,nstates,nx, nz);
            obj.gradabs = zeros(nstates,nstates,nx, nz);
            
            for s1 = 1:nstates;
               for s2 = 1:nstates; 
                    obj.gradx(s1,s2,:,:) = obj.delx(s1,:,:)-obj.delx(s2,:,:);
                    obj.gradz(s1,s2,:,:) = obj.delz(s1,:,:)-obj.delz(s2,:,:);
                    obj.gradabs(s1,s2,:,:) = sqrt(obj.gradx(s1,s2,:,:).^2 + obj.gradz(s1,s2,:,:).^2);
               end
            end

%             if ~isempty(obj.g_es)
%                 for it = 1:nsteps
%                         %theta = stepvec(it);
%                         theta = theta_vec(it);
%                         B0 = B_vec(1);
%                         B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;
% 
%                         HY1 = obj.beta*B*obj.g_es*obj.Sv3;
%                         HY1 = reshape(HY1,2,2);
%                         %HZ1_big = kron(HZ1,ones(2*spinI+1));
%                         if isempty(obj.Hhf_es)
%                             H_es =  kron(HY1,eye(2*obj.spinI+1));
%                         else
%                             H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
%                         end               
% 
%                         [V,D] = eig(H_es);
%                         obj.energies_es(it,:) = diag(D);
%                         obj.states_es(:,:,it) = V; % columns are the eigenvectors.
%                 end
%             end
%             
%             n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
%             n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
% 
%             obj.find_transitions;
            
            % calculate the transition strengths for the two polarizations.
            % 
            
        end
        
        function find_zeeman_gradient_angle(obj,thetax, Bz)
            % here, Bx now becomes an angle... Bz is magnitude
            % The goal here is to calculate the zeeman gradient at each
            % magnetic field. Essentially want to create a grid of magnetic
            % fields, which you'll input into B_x and B_z. I'm assuming a
            % uniaxial crystal here, so I'll apologize to my future self
            % for not making it more general!
            % I'm only thinking about the ground state spin transitions here, but you
            % could do a similar thing for the optical...
            % B_mat = [Bx By];
            nx = length(thetax);
            nz = length(Bz);
            
            obj.energies_gs = zeros(nx,2*(2*obj.spinI+1));
            obj.energies_es = zeros(nz,2*(2*obj.spinI+1));
            
            obj.states_gs = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nx,nz);
            obj.states_es = zeros(2*(2*obj.spinI+1),2*(2*obj.spinI+1),nz);
            
            obj.beta = obj.mub/(obj.h*obj.units); % to put levels in GHz.
            
            % Here I'll put the change in a given field level for perturbation
            obj.delx = zeros(2*(2*obj.spinI+1),nx,nz);
            obj.delz = zeros(2*(2*obj.spinI+1),nx,nz);
            
            nstates = 2*(2*obj.spinI+1);
            
            % my perturbations (i.e. what I need for the perturbation
            % theory) are 
           ptbx = obj.beta*[1,0,0]*obj.g_gs*obj.Sv3;
           ptbx = reshape(ptbx,2,2);
           ptbx = kron(ptbx,eye(2*obj.spinI+1));
           % disp(ptbx);
           
           ptbz = obj.beta*[0,0,1]*obj.g_gs*obj.Sv3;
           ptbz = reshape(ptbz,2,2);
           ptbz = kron(ptbz,eye(2*obj.spinI+1));
           
           obj.transx = zeros(nstates,nstates,nx, nz);
           obj.transz = zeros(nstates,nstates,nx, nz);
           
           obj.transitions_spin_gs = zeros(nstates,nstates,nx, nz);
            
            % the basic idea here is that I want to step over magnetic
            % fields and at each magnetic field calculate the gradient. 
            for itz = 1:nz
                for itx = 1:nx
                        %theta = stepvec(it);
                        %B0 = B_vec(it);
                        %B0 = B_vec(1); % in case I accidentally have a b vector. 
                        %theta = theta_vec(it);
                        B0 = Bz(itz);
                        theta = thetax(itx);
                        phi = 0;
                        B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)];
                        %B = [Bx(itx), 0, Bz(itz)];
                        
                        HZ1 = obj.beta*B*obj.g_gs*obj.Sv3;
                        HZ1 = reshape(HZ1,2,2);
                        
                        H_gs =  obj.Hhf_gs + kron(HZ1,eye(2*obj.spinI+1));
                        
                        %HZ1 = obj.beta*B*obj.g_es*obj.Sv3;
                        %HZ1 = reshape(HZ1,2,2);
                        
                        %H_gs =  obj.Hhf_es + kron(HZ1,eye(2*obj.spinI+1));
                        % calculate the eigenval and vectors at the given
                        % field
                        [V,D] = eig(H_gs);
                        % Columns of V are the eigenvectors;
                        obj.states_gs(:,:,itx,itz) = V;
                        obj.energies_gs(itz,:) = diag(D);
                        
                       % then to calculate the gradient, I first need the del for each transition
        
                       for state = 1:nstates;
                           obj.delx(state,itx,itz) = dot(V(:,state),ptbx*V(:,state));
                           obj.delz(state,itx,itz) = dot(V(:,state),ptbz*V(:,state));
                       end
                       
                       % calculate the transition probabilities
                       for s1 = 1:nstates;
                            for s2 = 1:nstates; 
                                obj.transx(s1,s2,itx,itz) = dot(V(:,s1),ptbx*V(:,s2));
                                obj.transz(s1,s2,itx,itz) = dot(V(:,s1),ptbz*V(:,s2));
                                obj.transitions_spin_gs(s1,s2,itx,itz) = D(s2,s2)- D(s1,s1);
                            end
                       end
                       
                end
            end
            
            % I then want to use this to calculate the gradient matrices.
            % i.e. the difference in the dels for a given transition. 
            
            obj.gradx = zeros(nstates,nstates,nx, nz);
            obj.gradz = zeros(nstates,nstates,nx, nz);
            obj.gradabs = zeros(nstates,nstates,nx, nz);
            
            for s1 = 1:nstates;
               for s2 = 1:nstates; 
                    obj.gradx(s1,s2,:,:) = obj.delx(s1,:,:)-obj.delx(s2,:,:);
                    obj.gradz(s1,s2,:,:) = obj.delz(s1,:,:)-obj.delz(s2,:,:);
                    obj.gradabs(s1,s2,:,:) = sqrt(obj.gradx(s1,s2,:,:).^2 + obj.gradz(s1,s2,:,:).^2);
               end
            end

%             if ~isempty(obj.g_es)
%                 for it = 1:nsteps
%                         %theta = stepvec(it);
%                         theta = theta_vec(it);
%                         B0 = B_vec(1);
%                         B = B0*[sin(theta)*cos(phi),    sin(theta)*sin(phi),    cos(theta)]+obj.B_offset;
% 
%                         HY1 = obj.beta*B*obj.g_es*obj.Sv3;
%                         HY1 = reshape(HY1,2,2);
%                         %HZ1_big = kron(HZ1,ones(2*spinI+1));
%                         if isempty(obj.Hhf_es)
%                             H_es =  kron(HY1,eye(2*obj.spinI+1));
%                         else
%                             H_es =  obj.Hhf_es + kron(HY1,eye(2*obj.spinI+1));
%                         end               
% 
%                         [V,D] = eig(H_es);
%                         obj.energies_es(it,:) = diag(D);
%                         obj.states_es(:,:,it) = V; % columns are the eigenvectors.
%                 end
%             end
%             
%             n_es = (2*obj.spinI + 1)*(2*obj.spinS + 1);
%             n_gs = (2*obj.spinI + 1)*(2*obj.spinS + 1);
% 
%             obj.find_transitions;
            
            % calculate the transition strengths for the two polarizations.
            % 
            
        end
    end
    
end

