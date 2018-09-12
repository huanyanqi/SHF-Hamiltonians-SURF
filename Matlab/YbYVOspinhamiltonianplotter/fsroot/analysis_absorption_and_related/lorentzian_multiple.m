function [ y ] = lorentzian_multiple( x, x0_vec, fwhm_vec, amp_vec  )
% function to plot multlple gaussian peaks.
%   npeaks is number of peaks. x0_vec is vector of peaks locations
%   fwhm is vector of full width half max for the peaks. amp_vec is a
%   vector of amplitudes
    npeaks = length(x0_vec);
    y = zeros(size(x));
    for peak = 1:npeaks
        %sigma = fwhm_vec(peak)/(2*sqrt(2*log(2)));
        %y = y + amp_vec(peak) * exp(-((x-x0_vec(peak)).^2)/(2 * sigma^2));
        y = y + (fwhm_vec(peak)/2)^2*amp_vec(peak)./((x-x0_vec(peak)).^2 + (fwhm_vec(peak)/2)^2);
    end



end

