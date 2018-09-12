set(findall(gcf,'-property','FontSize'),'FontSize',12)

% open fieldalongC_ramp_FFTdecay_combined_all_bgsub2_exclusions.fig;
% B = [1.0, 1.5, 3.0, 3.5, 4.0, 4.5] * 0.34; % Tesla
% RYbArr = [300,300,600,1000,1500,2100]; % Yb REI flip rate from the Bottger paper
% SensArr = [25.615E9, 25.289E9, 25.061E9, 25.041E9, 25.027E9, 25.018E9]; % Transition from 0(g) to 0(e) at various B fields
% TimeUnit = 1E-6;
% Temp = 0.65;

open RFspinechoT2_ramp_FFTdecay_combined_mimfits_bgsub2.fig;
B = [0.06, 0.24, 0.444]; % Tesla
RYbArr = [300,300,300]; % Yb REI flip rate from the Bottger paper
SensArr = [0.374E9, 0.0307E9, 0.0143E9]; % Transition from 0(g) to 1(g) at various B fields
TimeUnit = 1E-3;
Temp = 0.7;

h = findobj(gca,'Type','line');
x = get(h,'Xdata');
y = get(h,'Ydata');
N = length(x)/2;
close;

timeData = cell(N, 1);
echoData = cell(N, 1);

for i=1:N
    % Take 2i-1 to take only the odd series which are the experimental data
    % Divide time by 10^6 since microseconds.
    % Square the echo to get intensity from amplitude.
    timeData{i} = x{2*i-1}' * TimeUnit; 
    echoData{i} = (y{2*i-1}'.^2);
    
%     timeData{i} = timeData{i} - timeData{i}(1);
%     timeData{i} = timeData{i}(1:floor(length(timeData{i})/2));
%     echoData{i} = echoData{i}(1:floor(length(echoData{i})/2));
end

muB = 9.27E-24;
muN = 5.050784E-27;
h = 6.626E-34;
kB = 1.38E-23;
mu0 = pi*4E-7;
SYb = 1/2;
IYb = 1/2;
IY = 1/2;
IV = 7/2;

% Gamma = Gyromagnetic ratio for nuclear spin in units of Hz/T
gammaY = 2.1E6;
gammaV = 11.2E6;
% g factor, dimensionless. Can be converted to gamma by multiplying by the
% magneton (J/T) and then dividing by h (J/Hz).
gYbgEl = 6.08;
gYbNuc = 0.98734;
gammaYbEl = (gYbgEl*muB)/h;
gammaYbNuc = (gYbNuc*muN)/h;

% Estimation of number density of Y or V by generating a large file of Y
% and V atoms and counting the number of atoms and the distance of the
% furthest one.
neY = (33791/2)/(4/3*pi*(sqrt(56.95^2 + 56.95^2 + 50.31^2)*1E-10)^3);

% Spin flip rate using Bottger formula
RV = 0.25*mu0*h*gammaV^2*neY*sqrt(IV*(IV + 1));
RY = 0.25*mu0*h*gammaY^2*neY*sqrt(IY*(IY + 1));

% Magnetic field fluctuation from spin flips (with and without frozen core)
% Only the z component is included since B // z causing our energy levels 
% to only have gradient in the z direction
DeltaBYb = 0.66/1E4;
DeltaBV = (1.5*2.35)/1E4;
DeltaBY = (0.017*2.35)/1E4;
DeltaBYFrozen = (0.0017*2.35)/1E4;
DeltaBVFrozen = (0.0684*2.35)/1E4;

% List of models to fit the full I(t12) intensity decay of echo as
% function of delay time t12.
largeModel = fittype(@(I0, Gamma0, B, T, RYb, TransSens, t) ...
                        I0 - 4*t*pi.*(Gamma0 + ...
                        1/2 * DeltaBVFrozen*TransSens*RV*t + ...
                        1/2 * DeltaBYFrozen*TransSens*RY*t + ...
                        DeltaBYb*TransSens*sech(gYbgEl*muB*B/(2*kB*T))^2 * sqrt(2./(pi*RYb*t))), ...
             'problem', {'B', 'T', 'RYb', 'TransSens'}, 'independent', {'t'});
             
smallModelGamma0 = fittype(@(I0, Gamma0, B, T, RYb, TransSens, t) ...
                                I0 - 4*t*pi.*(Gamma0 + ...
                                1/2 * DeltaBVFrozen*TransSens*RV*t + ...
                                1/2 * DeltaBYFrozen*TransSens*RY*t + ...
                                1/2 * DeltaBYb*TransSens*RYb*t* sech((gYbgEl*muB*B)/(2*kB*T))^2), ...
                   'problem', {'B', 'T', 'RYb', 'TransSens'}, 'independent', {'t'});
               
smallModelGamma0SDV = fittype(@(I0, Gamma0, GammaSDIter, B, T, RYb, TransSens, t) ...
                               I0 * exp(- 4*t*pi.*(Gamma0 + ...
                                1/2 * GammaSDIter*RV*t + ...
                                1/2 * DeltaBYFrozen*TransSens*RY*t + ...
                                1/2 * DeltaBYb*TransSens*RYb*t* sech((gYbgEl*muB*B)/(2*kB*T))^2)), ...
                   'problem', {'B', 'T', 'RYb', 'TransSens'}, 'independent', {'t'});
               
smallModelGammaSDV = fittype(@(I0, GammaSDIter, B, T, RYb, TransSens, t) ...
                                I0 * exp(- 4*t*pi.*(1800 + ...
                                1/2 * GammaSDIter*RV*t + ...
                                1/2 * DeltaBYFrozen*TransSens*RY*t + ...
                                1/2 * DeltaBYb*TransSens*RYb*t*sech((gYbgEl*muB*B)/(2*kB*T))^2)), ...
                     'problem', {'B', 'T', 'RYb', 'TransSens'}, 'independent', {'t'});
                 
smallModelAll = fittype(@(I0, Gamma0, GammaSDIter, t) ...
                            I0 * exp(- 4*t *pi.*(Gamma0 + GammaSDIter*t)), ...
                'independent', {'t'});
            
MimsModel = fittype(@(I0, TM, x, t) ...
                            I0 * exp(- 2 * (2*t/TM).^x ), ...
                'independent', {'t'});

% Used to store the parameters for each plot after fitting
I0Arr = zeros(1,N);
Gamma0Arr = zeros(1,N);
GammaSDArr = zeros(1,N);
xArr = zeros(1,N);
TMArr = zeros(1,N);
cmap = hsv(N);
     
figure(1);
lines = zeros(1, N); 
markers = '+o*xsd';
hold on;

for i=1:N
    
%     f1 = fit(timeData{i}, echoData{i}, smallModelGammaSDV, ...
%             'StartPoint', [0, 1000000], ...
%             'Lower', [-Inf, 0], ...
%             'problem', {B(i), 0.65, RYbArr(i), 25*10^9}, ...
%             'Weights', exp([1:length(echoData{i})]*0.0));

%     [f1,gof] = fit(timeData{i}, echoData{i}, smallModelGamma0SDV, ...
%             'StartPoint', [0, 1800, 4E5], ...
%             'Lower', [0, 1000, 0], ...
%             'Upper', [Inf, 10000, Inf],...
%             'problem', {B(i), 0.65, RYbArr(i), 25*10^9}, ...
%             'Weights', exp([1:length(echoData{i})]*0.0));
        
%     f1 = fit(timeData{i}, echoData{i}, smallModelAll, ...
%             'StartPoint', [0, 1800, 100000], ...
%             'Lower', [-Inf, 0, 0],...
%             'Weights', exp([-1:-1:-length(echoData{i})]*-0.0));

    % Fit each series to the model with initial guesses and bounds
    % gof = goodness of fit
    [f1, gof] = fit(timeData{i}, echoData{i}, MimsModel, ...
            'StartPoint', [1, 1E-3, 2], ...
            'Lower', [0, 0, 0], ...
            'Upper', [Inf, 1E-2, 5]);
    disp(f1)
    disp(gof)

    % Store the fitted params
    xArr(i) = f1.x;
    TMArr(i) = f1.TM;

    plot(timeData{i}, feval(f1, timeData{i}), 'color', cmap(i, :), 'LineWidth', 2);
    h = plot(timeData{i}, echoData{i}, markers(i), 'color', cmap(i, :), 'MarkerSize', 10, 'DisplayName', strcat(string(B(i)), " T"));
    lines(i) = h;
    xlabel("t_{12} delay / s");
    ylabel("Photon Echo Intensity (a.u.)");
end
legend(lines)
set(gca, 'FontSize', 24)

% figure;
% plot(xArr, 'o-');5
% title("x Exponent");
% xlim([1 inf]); %ylim([0 inf]);

figure;
hold on;
plot(B, TMArr, 'o-', 'DisplayName','Experiment', 'MarkerSize', 10, 'LineWidth', 2);
% title("Phase Memory Time");
xlim([0 inf]); ylim([0 inf]);
xlabel("B field // c / T");
ylabel("Phase memory time / s");

% Used to store the predictions from the theory model
% Fast means the model that assumes R*t12 >> 1
predArr = zeros(1,N);
predArrFast = zeros(1,N);

% Compute the theoretical phase memory time
for i = 1:length(x)/2
   % Homogeneous linewidth assuming it's lifetime limited by T1
   % T1 ~ 270us, thus the width is 1/(pi*(2T1))
   a = 589; 
   % Formula from Bottger for the T_M
   b = 1/2 * DeltaBVFrozen * SensArr(i) * RV + 1/2 * DeltaBYb * SensArr(i) * RYbArr(i) * sech((gYbgEl*muB*B(i))/(2*kB*Temp))^2;
   predArr(i) = 1/b * (-a + sqrt(a^2 + 2*b/pi));
   % Formula from Zhong for the fast T_M
   predArrFast(i) = 2 * RYbArr(i) /(DeltaBYb * SensArr(i) * sech((gYbgEl*muB*B(i))/(2*kB*Temp))^2)^2;
end
plot(B, predArr, '*-', 'DisplayName','RT<1', 'MarkerSize', 10, 'LineWidth', 2)
plot(B, predArrFast, 'x-', 'DisplayName','RT>1', 'MarkerSize', 10, 'LineWidth', 2)
legend('Location','northwest');

MimsTmModel = fittype(@(a, DeltaBVFrozenIter, DeltaBYbIter, BB) ...
            1./(1/2 * DeltaBVFrozenIter * interp1q(B',SensArr',BB) * RV + 1/2 * DeltaBYbIter * interp1q(B',SensArr',BB) .* interp1q(B',RYbArr',BB) .* sech((gYbgEl*muB*BB)/(2*kB*Temp)).^2)...
            .* (-a + sqrt(a^2 + 2*(1/2 * DeltaBVFrozenIter * interp1q(B',SensArr',BB) * RV + 1/2 * DeltaBYbIter * interp1q(B',SensArr',BB) .* interp1q(B',RYbArr',BB) .* sech((gYbgEl*muB*BB)/(2*kB*Temp)).^2)/pi)),...
            'independent', {'BB'});
MimsTmFastModel = fittype(@(DeltaBYbIter, BB) ...
            2 * interp1q(B',RYbArr',BB) ./(DeltaBYbIter * interp1q(B',SensArr',BB) .* sech((gYbgEl*muB*BB)/(2*kB*Temp)).^2).^2,...
            'independent', {'BB'});

% [f2, gof] = fit(B', TMArr', MimsTmModel, ...
%             'StartPoint', [589, 7.63E-6, 1.02E-4], ...
%             'Lower', [-Inf, 0, 0], ...
%             'Upper', [Inf, 1E-4, 6E-4]);
% disp(f2)
% plot(B, feval(f2, B), 's-', 'DisplayName', 'Fitted RT<1', 'MarkerSize', 10, 'LineWidth', 2)

[f3, gof] = fit(B', TMArr', MimsTmFastModel, ...
            'StartPoint', [9.8E-5], ...
            'Lower', [0], ...
            'Upper', [6E-4]);
disp(f3)
plot(B, feval(f3, B), 's-', 'DisplayName', 'Fitted RT>1', 'MarkerSize', 10, 'LineWidth', 2)

set(gca, 'FontSize', 24)
