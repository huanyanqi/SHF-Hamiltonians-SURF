% all of the values are now defined in the function below in an aspittempt to ensure consistency across me scripts:             
ybyvo = yb171yvospinhamiltonian;

%% step over magnetic field amplitudes for a fixed angle
ybyvo.B_offset = [0 0 0]; % offset field x y z (in T)
anglewrtC = 90; % in degrees
bvec = 0.0:0.001:0.360;
%ybyvo.step_magnetic_field_amplitude(0.0277:0.000001:0.0278,0*pi/180,0) % field amplitude, theta, phi. angles in radians
ybyvo.step_magnetic_field_amplitude(bvec,anglewrtC*pi/180,0) % field amplitude, theta, phi. angles in radians
% plot ground and excited energies; 

%[fig_h,axes_gs, axes_es] = ybyvo.plot_energies_all([],[],[]); 
% optional arguments are for figure/axes handl
% can also output these if so desired.


%ylim(axes_gs,[-5,5])
%ylim(axes_es,[-5,5])
%title(axes_gs,[sprintf('ground state energies. B at %3.2f',anglewrtC) '\circ w.r.t C'])
%title(axes_es,[sprintf('excited state energies. B at %3.2f',anglewrtC) '\circ w.r.t C'])

[~,axes_gs_split,axes_es_split] = ybyvo.plot_splittings_all([],[],[]); 
title(axes_gs_split,[sprintf('ground state splitting. B at %3.2f',anglewrtC) '\circ w.r.t C'])
title(axes_es_split,[sprintf('excited state splitting. B at %3.2f',anglewrtC) '\circ w.r.t C'])
%)
ylim(axes_gs_split,[0,5])
ylim(axes_es_split,[0,5])
%%
ybyvo.find_transitions_spin
figure;
subplot(2,1,1)
plot(ybyvo.B_vec,ybyvo.transitions_spin_es)
title('all es transitions')
ylim([0 12])
subplot(2,1,2)
plot(ybyvo.B_vec,ybyvo.transitions_spin_gs)
title('all gs transitions')
ylim([0 25])

%%
% bs = ybyvo.B_vec(2)-ybyvo.B_vec(1); 
% 
% gs_split = ybyvo.energies_gs(:,2:end)-ybyvo.energies_gs(:,1:end-1);
% grad = (gs_split(2:end,:) - gs_split(1:end-1,:))/bs;
% %hold on
% %plot(ybyvo.B_vec(1:end-1),gs_split(2:end,:) - gs_split(1:end-1,:))
% figure;
% plot(ybyvo.B_vec(1:end-1),1000*grad)
% ylim([-50,50])
% ylabel('df/dB (MHz/T)')
% xlabel('B (T)')
%% calculate transitions and plot 

[~,axes_alltrans] = ybyvo.plot_transitions_all([],[]);

ylim(axes_alltrans,[-15,15])
%title(axes_alltrans,[sprintf('YbYVO. All transitions. B %3.2f',anglewrtC) '\circ w.r.t C'])
xlim(axes_alltrans,[0,max(ybyvo.B_vec)])
xlabel(axes_alltrans,'B (T)')
ylabel(axes_alltrans,'Detuning (GHz)')

% %% calculate transitions and plot 
% ndyvo.plot_transitions_all([],[]);
%%
figure;
p = plot(ybyvo.B_vec,ybyvo.transitions);
c = colormap('lines');
%%
for it = 1:4
    p(it).Color = c(1,:);
end



