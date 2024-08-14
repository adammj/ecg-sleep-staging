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


function [local_variation] = calculate_local_variation(input, local_samples, group_samples, nearly_zero_value)
%get the local variation:
%  moving variance of a small group divided by a moving median of the same
%  moving variance of a larger group
%
% assumptions:
%  the array is column-major (vertically oriented), but can have multiple columns
%  local_width and group_width are in seconds
%  clip_below is a fixed value, with 1e-10 being a good start

assert(size(input,1) >= size(input,2), 'input must column-major (vertically oriented)');

if nargin < 4
    %clip to one all of the ratios that could be erroneous
    %this helps prevent almost zero values from growing (by dividing by zero)
    nearly_zero_value = 1e-10;

    if nargin < 3
        %the moving median of the standard deviations
        group_samples = max(5, round(length(input)/10));
        
        if nargin < 2
            %the width for the standard deviation
            local_samples = max(3, round(length(input)/200));
        end
    end
end

%calculate local sample lengths
local_samples = next_odd(local_samples);
group_samples = next_odd(group_samples);

%make sure the group is larger than the local
if group_samples < (local_samples*3 + 1)
    group_samples = (local_samples*3 + 1);
end
assert(group_samples > local_samples, 'group samples must be greater than local samples')

%calculate the near and wide arrays
local_array = movvar(input, local_samples, 'omitnan');
group_array = movmedian(local_array, group_samples, 'omitnan');
local_variation = local_array./group_array;

%remove the erroneous values that could go to infinity by clipping them to 0
local_variation(local_array < nearly_zero_value | group_array < nearly_zero_value) = 0;

%clip out any NaNs
local_variation(isnan(local_variation)) = 0;

%scale the output using log (maintaining 0=0, and 1=1)
local_variation = log(local_variation + 1)/log(2);

%check output
assert(min(local_variation) >= 0)

end
