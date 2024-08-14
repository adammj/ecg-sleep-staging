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


function [ecg_output, per_mad_masked, per_max_masked, indx_mad, indx_max, mask_mad, mask_max] = normalize_ecg(ecg, sec_to_use, fs)

assert(isvector(ecg))
assert(size(ecg, 1) > 1)

%% first, remove the median
ecg = ecg - median(ecg);

%% find the windows that have larger than expected MAD
temp_ecgs = reshape(ecg, [fs*sec_to_use], [])';
mad_values = mad(temp_ecgs', 1)';
desired_mad = 3 * mad(ecg, 1);
indx_mad = find(mad_values > desired_mad);

per_mad_masked = length(indx_mad)/size(temp_ecgs,1);
% fprintf('mad masked: %f\n', per_mad_masked);

% create mask
mask_mad = ones(size(temp_ecgs, 1), 1);
mask_mad(indx_mad) = desired_mad ./ mad_values(indx_mad);

% expand mask
mask_mad = repelem(mask_mad, (fs*sec_to_use));

%
% need to smooth the mask here
%
mask_mad = smooth_mask_array_nonbinary(mask_mad, fs*sec_to_use/2);

% apply mask
ecg_output = (ecg - median(ecg)) .* mask_mad;



%% Do the same, but now for the max abs value of a window
temp_ecgs = reshape(ecg_output, [fs*sec_to_use], [])';
max_values = max(abs(temp_ecgs'), [], 1)';
desired_max = 2 * prctile(max_values, 95);
indx_max = find(max_values > desired_max);

per_max_masked = length(indx_max)/size(temp_ecgs,1);
% fprintf('max masked: %f\n', per_max_masked);

mask_max = ones(size(temp_ecgs, 1), 1);
mask_max(indx_max) = desired_max ./ max_values(indx_max);

% expand mask
mask_max = repelem(mask_max, (fs*sec_to_use));

%
% need to smooth the mask here
%
mask_max = smooth_mask_array_nonbinary(mask_max, fs*sec_to_use/2);

% apply mask
ecg_output = (ecg_output - median(ecg_output)) .* mask_max;



%% normalize (and scale down to 1/10th, to give approx +/-1)
ecg_output = ((ecg_output - median(ecg_output))./mad(ecg_output, 1))/10;


end
