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


function [mask_array] = fill_mask_array_gaps(mask_array, max_gap_length)
%fill in gaps between sections

count = 0;
cycle_start = 1;
for i = 1:length(mask_array)
    if mask_array(i) == 1
        count = count + 1;
    else
        if count > 0 
            % if the length isn't too long, fill in gap
            if count <= max_gap_length
                mask_array(cycle_start:i) = 0;
            end
            count = 0;
        end
        cycle_start = i;
    end 
end

end


