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


function [acf, lags] = autocorr_merged(y, lag_bounds, clip_negative)
% normalized autocorrelation, merging fixed and variable width windows
% the bounds are a vector of 3 numbers
%     1) smallest lag
%     2) end of fixed_width window lag (variable_width begins on the next lag)
%     3) largest lag
% window is always be taken from the middle of the timeseries

%set default inputs
if nargin < 3
    clip_negative = true;
end

%check inputs
assert(isvector(lag_bounds));
assert(isvector(y), 'y must be a vector');
if ~iscolumn(y)
    y = y'; %converted to a column, so that the acf and lags come out as columns
end
assert(length(lag_bounds) == 3, 'lag bounds must be length 3');
assert(lag_bounds(1) >= 0, 'lag_bounds(1) cannot be less than 0');
assert(lag_bounds(2) >= lag_bounds(1), 'lag_bounds(2) must be >= lag_bounds(1)');
assert(lag_bounds(3) >= lag_bounds(2), 'lag_bounds(3) must be >= lag_bounds(2)');
assert(length(y) >= 2*lag_bounds(3), 'length(y) must be 2*lag_bounds(3)');
assert(sum(mod(lag_bounds,1) == 0) == 3, 'lag_bounds must all be integers')

%don't actually calculate a lag of 0 (always 1)
if lag_bounds(1) == 0
    acf_zero = 1;
    lags_zero = 0;
    lag_bounds(1) = 1;
    if lag_bounds(2) == 0
        lag_bounds(2) = 1;
    end
    if lag_bounds(3) == 0
        lag_bounds(3) = 1;
    end
else
    acf_zero = [];
    lags_zero = [];
end

%get the lower part of the acf (fixed_width)
[acf_lower, lags_lower] = autocorr_fixed(y, lag_bounds(1:2), clip_negative);

%get the upper part of the acf (variable_width)
if lag_bounds(3) > lag_bounds(2)
    [acf_upper, lags_upper] = autocorr_variable(y, [lag_bounds(2)+1, lag_bounds(3)], clip_negative);
else
    acf_upper = [];
    lags_upper = [];
end

%combine the outputs
acf = [acf_zero; acf_lower; acf_upper];
lags = [lags_zero; lags_lower; lags_upper];

end
