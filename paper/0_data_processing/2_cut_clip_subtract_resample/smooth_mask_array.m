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


function [mask_array] = smooth_mask_array(mask_array, transition_length)
%smooth the mask array transitions with a spine fit
%the transition_length is the number of values to be between 0 and 1
%assumptions:
% 0 is where the mask is clipping (left alone)
% 1 is where the mask is not clipping
% the smoothing will cut into the non-clipped area
% if the mask begins or ends with a 1, regardless of transition, it will
% remain so

assert(transition_length > 0)
assert(isvector(mask_array))
assert(length(mask_array) > (2+transition_length))
num_unique_values = length(unique(mask_array));

%if there are no clips, exit
if num_unique_values == 1
    return;
end
assert(num_unique_values == 2)


%smooth spline, with end derivatives fixed at 0
spline_pp = spline([0, 0.5, 1],[0, 0, 0.5, 1, 0]);

%smoothly transition the clips both on and off
transistion_on_i = find(mask_array(1:end-1) == 1 & mask_array(2:end) == 0) + 1;
transisiton_off_i = find(mask_array(1:end-1) == 0 & mask_array(2:end) == 1);

%fix the transitions on
for i = 1:length(transistion_on_i)
    smoooth_i = transistion_on_i(i);
    smooth_beg_i = smoooth_i - (transition_length + 1);
    if smooth_beg_i < 1
        smooth_beg_i = 1;
    end
    
    clip_len = smoooth_i - smooth_beg_i + 1;
    clip_values = (1:-1/(clip_len-1):0)';
    
    %smooth out the values
    clip_values = ppval(spline_pp, clip_values);
    mask_array(smooth_beg_i:smoooth_i) = mask_array(smooth_beg_i:smoooth_i).*clip_values;
end

%fix the transitions off
for i = 1:length(transisiton_off_i)
    smoooth_i = transisiton_off_i(i);
    smooth_end_i = smoooth_i + (transition_length + 1);
    if smooth_end_i > length(mask_array)
        smooth_end_i = length(mask_array);
    end
    
    clip_len = smooth_end_i - smoooth_i + 1;
    clip_values = (0:1/(clip_len-1):1)';
    
    %smooth out the values
    clip_values = ppval(spline_pp, clip_values);
    mask_array(smoooth_i:smooth_end_i) = mask_array(smoooth_i:smooth_end_i).*clip_values;
end

end
