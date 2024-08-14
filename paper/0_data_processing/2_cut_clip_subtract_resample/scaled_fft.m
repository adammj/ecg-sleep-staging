% Copyright (C) 2024  Adam Jones  All Rights Reserved
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.


function [ratio_P, f] = scaled_fft(input_full, fs, freq_div, min_value)
%get the scaled fft (value/median)

if nargin < 4
    min_value = 0.01;
end

assert(isvector(input_full))
assert(isscalar(fs))

%minimum of 5 bins, 7 for 1Hz divs, 13 for 0.5Hz divs
%these values were empirically tested to work well
frequency_bins = max(next_odd(6/freq_div + 1), 5);  

%get just the input window necessary
input = input_full(1:round(fs/freq_div));

fft_len = length(input);
Y = fft(input,fft_len);
abs_Y = abs(Y); %no need to divide by length
ratio_P = abs_Y(1:round(fft_len/2)+1);
ratio_P(2:end-1) = 2*ratio_P(2:end-1);
f = fs*(0:round(fft_len/2))'/fft_len;

%scale the fft values
fft_median_val = movmedian(ratio_P, frequency_bins, 'omitnan');
fft_median_val(fft_median_val < min_value) = min_value;

ratio_P = ratio_P./fft_median_val;

%correct any inf values
ratio_P(isinf(ratio_P)) = 1;

end
