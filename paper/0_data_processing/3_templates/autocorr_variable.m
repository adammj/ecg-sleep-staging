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


function [acf, lags] = autocorr_variable(y, lag_bounds, clip_negative)
% normalized autocorrelation based on:
% https://gerrybeauregard.wordpress.com/2013/07/15/high-accuracy-monophonic-pitch-estimation-using-normalized-autocorrelation/
% normalizes all lags, so that value = 1 means perfect periodicity
% negative correlations are clipped to 0 by default
% by default, gives the output for 1 to floor(length(y)/2)
% window is always be taken from the middle of the timeseries


%set default inputs
if nargin < 3
    clip_negative = true;   

    if nargin < 2
        lag_bounds = [1, floor(length(y)/2)];  %default to half the length
    end
end

%check inputs
assert(isvector(lag_bounds));
assert(isvector(y));
assert(length(lag_bounds) == 2, 'lag bounds must be length 2');
assert(lag_bounds(1) >= 1, 'lag_bounds(1) cannot be less than 1 for variable width');
assert(lag_bounds(2) >= lag_bounds(1), 'lag_bounds(2) must be >= lag_bounds(1)');
assert(length(y) >= 2*lag_bounds(2), 'length(y) must be at least 2*lag_bounds(2)');
assert(islogical(clip_negative), 'clip_negative must be logical');
assert(sum(mod(lag_bounds,1) == 0) == 2, 'lag_bounds must all be integers')

%get the window
y = select_centered_window(y, 2*lag_bounds(2));

%set up output arrays (lags, acf)
lags = (lag_bounds(1):lag_bounds(2))';
acf = zeros(length(lags), 1, 'single');

%calculate start and end indexes
center_index = length(y)/2;     %update center index
start_indexes = center_index - lags + 1;
end_indexes = center_index + lags;

%precalculate the mean
before_center_cumsum = cumsum(flip(y(1:center_index))); %flipped to sum from center-out
after_center_cumsum = cumsum(y((center_index+1):end));  %sum from center-out
precalc_mean = (after_center_cumsum + before_center_cumsum)./(2*(1:length(before_center_cumsum))');
precalc_mean(1:(lags(1)-1)) = [];  %erase unused lags

%calculate each correlation for each lag (variable window)
%note: there doesn't seem to be a great way to vectorize this loop, without
%performing an order of maginitude more operations
for lag_i = 1:length(lags)
    lag = lags(lag_i, 1);

    %the mean must be removed for each individual lag
    y_local = y(start_indexes(lag_i):end_indexes(lag_i)) - precalc_mean(lag_i);
    
    ac = sum(y_local(1:lag) .* y_local((lag+1):end));
    
    %negative correlations are ignored
    if (ac > 0) || ~clip_negative
        sum_sq_beg = sum(y_local(1:lag).^2);
        sum_sq_end = sum(y_local((lag+1):end).^2);
        acf(lag_i, 1) = ac*abs(ac)/(sum_sq_beg*sum_sq_end);   %correct for sign
    end
end

end
