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


function [signal, clip_array, mask_array, mask_count] = remove_large_clips(signal, clip_length, clip_ratio, fill_gap_length, transition_length)
%looks for areas with above-average clipped signal (always occurs where
%square and pseudo-square waves are)
%will assume small gaps between detected square waves should also be removed
%smooths the transitions around the clipped sections

%make sure input is vertical
assert(iscolumn(signal))

%calculate clip array
clip_array = get_clip_array(signal);

%set constants
mask_array = ones(size(signal), 'single');

%look for above-average clip ratio (looking right and left)
avg_clip_right = movmean(abs(clip_array), [0, clip_length - 1], 'omitnan');
avg_clip_left = movmean(abs(clip_array), [clip_length - 1, 0], 'omitnan');

mask_array((avg_clip_right >= clip_ratio) | (avg_clip_left >= clip_ratio)) = 0;

%fill in small gaps
mask_array = fill_mask_array_gaps(mask_array, fill_gap_length);

%smooth the mask array transitions
mask_array = smooth_mask_array(mask_array, transition_length);

%get the mask count (where the mask is exactly 0)
mask_count = length(find(mask_array == 0));

%apply the mask to the signal
%get median of non-masked portions
median_value = median(signal.*mask_array, 'omitnan');
%subtract out median, and apply mask
%but don't add median back in
signal = (signal - median_value).*mask_array;

end
