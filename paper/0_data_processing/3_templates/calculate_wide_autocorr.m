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


function [autocorr_wide] = calculate_wide_autocorr(autocorr, constants)
%smooth the autocorr_wide to better detect the center of the range

%to prevent scaling up noise
min_scale_floor = constants.min_peak_corr * 2;

autocorr_wide = movmean(autocorr, constants.wide_width, 'omitnan');

%scale the area of interest
start_i = constants.bounds(2) + 1;
autocorr_wide(:,start_i:end) = autocorr_wide(:,start_i:end) ./ max(min_scale_floor, movmax(max(autocorr_wide(:,start_i:end), [], 2, 'omitnan'), constants.wide_max_scale_width, 'omitnan'));

end
