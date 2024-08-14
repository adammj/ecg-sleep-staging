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


function [clip_array] = get_clip_array(signal)
%get the clipped indicies (useful for the remove bounce, or the r peak
%finding)

%get upper and lower targets (1 delta in from max and min)
upper_target = max(signal); % - smallest_delta;
lower_target = min(signal); % + smallest_delta;

%old method
%TARGET_PERCENT = 0.01;  %find the values at this percent of highest and lowest values
%sorted_signal = sort(signal);
%upper_target = sorted_signal(round(length(sorted_signal)*(1-TARGET_PERCENT)));
%lower_target = sorted_signal(round(length(sorted_signal)*TARGET_PERCENT));
%num_deltas = (upper_target - lower_target) / smallest_delta;
%upper_target = upper_target - num_deltas/10*smallest_delta; %bring it in 10%
%lower_target = lower_target + num_deltas/10*smallest_delta; %bring it in 10%


%get array of upper and lower clips
clip_array = zeros(size(signal), 'single');
clip_array(signal >= upper_target) = 1;
clip_array(signal <= lower_target) = -1;


end

