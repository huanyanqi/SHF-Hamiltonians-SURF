%% initialize the spin hamiltonian objects
ybyvo171 = yb171yvospinhamiltonian;
ybyvo173 = yb173yvospinhamiltonian;
ybyvo172 = yb170yvospinhamiltonian;


%% 
% first, i'm plotting the  transition strengths for the 171 isotope with 
%E || c
%stepvec = 0;
bvec = 0:0.001:0.1;
%ybyvo.find_branching_ratio_relative(bvec,10*pi/180,0)
%ybyvo.find_transition_strengths_fieldangle(bvec,90*pi/180,0)
anglewrtc = 0;
ybyvo171.find_transition_strengths_amplitude(bvec,anglewrtc*pi/180,0)
ybyvo173.find_transition_strengths_amplitude(bvec,anglewrtc*pi/180,0)
ybyvo172.find_transition_strengths_amplitude(bvec,anglewrtc*pi/180,0)


x = -20:0.001:20;
widths171 = 0.25*ones(length(ybyvo171.transition_strengths_par(1,:)),1);
heights171 = ones(length(ybyvo171.transition_strengths_par(1,:)),1);

widths173 = 0.25*ones(length(ybyvo173.transition_strengths_par(1,:)),1);
heights173 = ones(length(ybyvo173.transition_strengths_par(1,:)),1);
widths172 = 0.25*ones(length(ybyvo172.transition_strengths_par(1,:)),1);
heights172 = ones(length(ybyvo172.transition_strengths_par(1,:)),1);
%
% figure; 
% ha1 = axes; %subplot(2,1,1);
% %ha2 = subplot(2,1,2);
% hold(ha1,'on')
% %hold(ha2,'on')
% for it = 1;%:length(bvec)  
%     plot(ha1,x,lorentzian_multiple( x, ybyvo.transitions(it,:), widths, ybyvo.transition_strengths_par(it,:)) )
%     plot(ha1,x,lorentzian_multiple( x, ybyvo.transitions(it,:), widths, ybyvo.transition_strengths_perp(it,:)) )
%     plot(ha1,x,lorentzian_multiple( x, ybyvo.transitions(it,:), widths, 0.2*heights),'--' )
% end
% hold(ha1,'off')
% %hold(ha2,'off')
% xlabel('detuning (GHz)')
% %drawnow

% making a matrix of this and doing a surf plot. 
parmat = zeros(length(bvec),length(x));
parmat0spin = zeros(length(bvec),length(x));
for it = 1:length(bvec)
    parmat(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, ybyvo171.transition_strengths_par(it,:))));
    %parmat0spin(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo0spin.transitions(it,:), widths, ybyvo0spin.transition_strengths_par(it,:))));
    %perpmat(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo.transitions(it,:), widths, 0.2*heights)));
end
%
figure('name','new: BperpC, EparC');
%surf(bvec*1000, x, perpmat','linestyle','none')
imagesc(bvec*1000, x, parmat',[0 1])
%hold on
%plot(bvec*1000,ybyvo.transitions)
set(gca,'YDir','normal')
c = bone;
%c = flipud(c);
colormap(c);
%view([0,90])
xlabel('B \perp C (mT)')
ylabel('Detuning (GHz)')

%%
% here I'm plotting multiple isotopes on top of each other
it = 1;
mat171 = (lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, ybyvo171.transition_strengths_par(it,:)));
mat173 = (lorentzian_multiple( x, ybyvo173.transitions(it,:), widths173, ybyvo173.transition_strengths_par(it,:)));
mat172 = (lorentzian_multiple( x, ybyvo172.transitions(it,:), widths172, ybyvo172.transition_strengths_par(it,:)));%figure;
figure;
%plot(x,0.14*mat171/4,x,mat172,x,0.16*mat173/12)
plot(x,mat171,x,mat172,x,mat173)
legend('171','even isotopes','173')
legend('boxoff')
legend({},'FontSize',24)

xlim([-6,6])
xlabel('detuning (GHz)')
title('Allowed transitions for E || c')
ylabel('relative transition strength ')
% then taking into account relative natural abundance
maxstrength = max(0.14*mat171/4 + 0.6*mat172/2 + 0.16*mat173/12);
figure;
hold on
plot(x,0.14*mat171/4/maxstrength,x,0.6*mat172/2/maxstrength,x,0.16*mat173/12/maxstrength)
plot(x,(0.14*mat171/4 + 0.6*mat172/2 + 0.16*mat173/12)/maxstrength)
hold off
legend('171','even isotopes','173','summed')
legend({},'FontSize',24)
legend('boxoff')
xlim([-6,6])
title('Accounting for relative abundance')
xlabel('detuning (GHz)')
ylabel('relative transition strength ')
%%
ix = 2*ybyvo171.bigI(:,1:4);
iz = 2*ybyvo171.bigI(:,9:12);
sz = 2*ybyvo171.bigS(:,9:12);

i_gs = zeros(4,1);
i_es = zeros(4,1);
s_gs = zeros(4,1);
s_es = zeros(4,1);

for it = 1:4
    i_gs(it) =  ybyvo171.states_gs(:,it,end)'*iz*ybyvo171.states_gs(:,it,end);
    i_es(it) =  ybyvo171.states_es(:,it,end)'*iz*ybyvo171.states_es(:,it,end);

    s_gs(it) =  ybyvo171.states_gs(:,it,end)'*sz*ybyvo171.states_gs(:,it,end);
    s_es(it) =  ybyvo171.states_es(:,it,end)'*sz*ybyvo171.states_es(:,it,end);
end


%% testing out the transition strengths for the two different directions.

%stepvec = 0;
bvec = 0.400;
theta_vec = [1];
ybyvo171.B_offset = [0 0 0];
ybyvo171.find_transition_strengths_fieldangle(bvec,theta_vec*pi/180,0)


branching_ratios_par = ybyvo171.find_branching_ratio_fieldangle(bvec,theta_vec*pi/180,0, [2],[1 3]);
% disp(ybyvo.energies_gs(1,:))
% disp(ybyvo.states_gs(:,:,1))
% figure;
% plot(stepvec,ndyvo.R)
% 
% beta1 = 1./(1+ndyvo.R);
% beta2 = (ndyvo.R)./(1+ndyvo.R);
% 
% figure;
% plot(stepvec,beta1,stepvec,beta2)
x = -50:0.001:50;
widths171 = 0.05*ones(length(ybyvo171.transition_strengths_par(1,:)),1);
heights171 = ones(length(ybyvo171.transition_strengths_par(1,:)),1);

widths173 = 0.05*ones(length(ybyvo173.transition_strengths_par(1,:)),1);
heights173 = ones(length(ybyvo173.transition_strengths_par(1,:)),1);
widths172 = 0.05*ones(length(ybyvo172.transition_strengths_par(1,:)),1);
heights172 = ones(length(ybyvo172.transition_strengths_par(1,:)),1);

figure; 
ha1 = axes; %subplot(2,1,1);
%ha2 = subplot(2,1,2);
hold(ha1,'on')
%hold(ha2,'on')
for it = 1:length(theta_vec)  
    plot(ha1,x,lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, ybyvo171.transition_strengths_par(it,:)) )
    plot(ha1,x,lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, ybyvo171.transition_strengths_perpx(it,:)) )
    plot(ha1,x,lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, 0.2*heights171),'--' )
end
hold(ha1,'off')
%hold(ha2,'off')
xlabel('detuning (GHz)')
legend('E || C', 'E \perp C', 'all lines' )
% %%
% bvec = 0.1;
% theta_vec = [0:1:90];
% 
% ybyvo171.find_transition_strengths_fieldangle(bvec,theta_vec*pi/180,0)
% % I guess if your field gets too low, you'll start tracking the wrong
% % states! good idea to also plot the transitions...
% branching_ratios_1 = ybyvo171.find_branching_ratio_fieldangle(bvec,theta_vec*pi/180, 0, [1],[3 4]);
% branching_ratios_2 = ybyvo171.find_branching_ratio_fieldangle(bvec,theta_vec*pi/180, 0, [2],[3 4]);
% %branching_ratios_3 = ybyvo.find_branching_ratio_fieldangle(bvec,theta_vec*pi/180,0, [3],[1 2]);
% %5branching_ratios_4 = ybyvo.find_branching_ratio_fieldangle(bvec,theta_vec*pi/180,0, [4],[1 2]);
% figure;
% ha = axes;
% hold(ha,'on')
% plot(ha,theta_vec,branching_ratios_1,'-')
% %plot(ha,theta_vec,branching_ratios_2,'-.')
% hold(ha,'off')
% %%
% 
% bvec = 0.01;
% theta_vec = [0:0.1:90];
% x = -30:0.001:30;
% ybyvo171.find_transition_strengths_fieldangle(bvec,theta_vec*pi/180,0)
% 
% 
% parmat = zeros(length(theta_vec),length(x));
% for it = 1:length(theta_vec)
%     parmat(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo171.transitions(it,:), widths171, ybyvo171.transition_strengths_par(it,:))));
%     %parmat0spin(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo0spin.transitions(it,:), widths, ybyvo0spin.transition_strengths_par(it,:))));
%     %perpmat(it,:) = exp(-5*(lorentzian_multiple( x, ybyvo.transitions(it,:), widths, 0.2*heights)));
% end
% %
% figure();
% %surf(bvec*1000, x, perpmat','linestyle','none')
% imagesc(theta_vec, x, parmat')
% set(gca,'YDir','normal')
% c = bone;
% %c = flipud(c);
% colormap(c);
% %view([0,90])
% %xlabel('B \perp C (mT)')
% ylabel('Detuning (GHz)')
% 
% % bvec = 0.4;
% % anglevec = [0:1:90]; 
% % figure
% % ha3 = axes;
% % figure;
% % ha4 = axes;
% % hold(ha3,'on')
% % hold(ha4,'on')
% % for angle = anglevec;
% %     %ybyvo.find_branching_ratio_relative(bvec,angle*pi/180,0)
% %     ybyvo.find_branching_ratio_relative_subset(bvec,angle*pi/180,0)
% %     ha3.ColorOrderIndex = 1;
% %     ha4.ColorOrderIndex = 1;
% %     plot(ha3,angle,ybyvo.transition_strengths_ll/sum(ybyvo.transition_strengths_ll),'o')
% %     %plot(ha3,angle,ybyvo.transition_strengths_ll(:,2:-1:1)./ybyvo.transition_strengths_ll,'o')
% %     plot(ha4,angle,ybyvo.transitions(:,[5:6]),'o')
% %     plot(ha4,angle,ybyvo.transitions(:,[1:2]),'x')
% % end
% % ylim(ha3,[0,5])
% % hold(ha4,'off')
% % hold(ha4,'off')