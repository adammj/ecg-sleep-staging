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


function [local_variation] = filter_local_variation(local_variation, lv_lp_filter)
%low-pass filter the local variation, to help emphasize peaks

%find where the local variation array is already 0 (filter can add noise to
%completely silent sections)
local_variation_clipped = ones(size(local_variation), class(local_variation));
local_variation_clipped(local_variation <= 0) = 0;

%filter the local variation array
local_variation = filtfilt(lv_lp_filter, double(local_variation));

%reset all values that were 0, to 0
local_variation = local_variation .* local_variation_clipped;

%don't allow any value to be below 0
local_variation(local_variation < 0) = 0;

end
