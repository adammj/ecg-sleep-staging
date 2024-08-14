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


function [autocorr] = filter_autocorr_harmonics(autocorr, bounds, plus_minus_lag)
%de-emphasizes higher harmonics by using the correlation of the lower
%harmonic as the scale
%if lower harmonic = 0.85, then scale higher harmonic by (1-0.85)=0.15
%
%bounds are (lowest, highest) index of interest
%assumes autocorr is of shape (epochs, lags)

if nargin < 3
    plus_minus_lag = 0;
end

assert(isvector(bounds))
assert(length(bounds) == 2)

max_harmonic = ceil(bounds(2)/bounds(1));

for current_i = bounds(2):-1:(bounds(1)*2)
    %start from 1/2 harmonic, down to 1/max harmonic
    for harmonic_div = 2:max_harmonic
        harmonic_i = round(current_i/harmonic_div);
        start_i = harmonic_i - plus_minus_lag;
        end_i = harmonic_i + plus_minus_lag;
        
        %test that the lag is still in the range desired
        if harmonic_i > bounds(1)
            autocorr(:, round(current_i)) = (1 - max(autocorr(:, start_i:end_i), [], 2)) .* autocorr(:, round(current_i));
        end
    end
end

end
