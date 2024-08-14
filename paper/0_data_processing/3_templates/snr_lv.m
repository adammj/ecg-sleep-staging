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


function [output] = snr_lv(input, fs, lv_peaks)
% gives a rough measure of the noise around lv peaks
% from 0 to 10
% constants were chosen by trial and error to work on a wide-range
% of noisy data, and to separate good from noisy peaks


window_1 = next_odd(fs*0.6);  % just local to peak
window_2 = next_odd(fs*2);    % surrounding area
ratio = 0.5;  % ratio of peak height to surrounding area

% initial function
output = 10*movmean(input > (movmax(input, window_1)*ratio), window_2);


% adjustment for the peak height
noise_for_peaks = output(lv_peaks(:,1));
noise_for_peaks = min(10, max(0, noise_for_peaks - 0.002*max(0, (lv_peaks(:, 2)-1.5)).^4));

output = noise_for_peaks;

end

function [x] = next_odd(x)
%gives next odd integer up

x = 2*floor(ceil(x)/2) + 1;

end