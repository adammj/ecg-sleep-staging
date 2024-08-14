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


function [autocorr] = filter_autocorr_harmonics_new(autocorr)

min_pk = 0.1;
lower_lag = 77;
plusminus = 30;

max_len = size(autocorr, 2);

for ac_i = 1:size(autocorr,1)

    ac_slice = autocorr(ac_i, :);
    
    if max(ac_slice(lower_lag:end)) < min_pk
        continue
    end

    [pks,locs] = findpeaks(ac_slice, 'MinPeakHeight', min_pk);
    pks(locs < lower_lag) = [];
    locs(locs < lower_lag) = [];
    
    for loc_i = 1:length(locs)
        % source_start_i = locs(loc_i) - plusminus;
        % source_end_i = min(length(ac_slice), locs(loc_i) + plusminus);
        
        % source_scale = 1 - ac_slice(source_start_i:source_end_i);
        source_scale = 1 - pks(loc_i);
        % source_len = source_end_i - source_start_i + 1;
    
        for harmonic_i = 2:6
            dest_start_i = locs(loc_i)*harmonic_i - plusminus;
            if dest_start_i > max_len
                break
            end
            dest_end_i = min(max_len, locs(loc_i)*harmonic_i + plusminus);
            
            % dest_len = dest_end_i - dest_start_i + 1;
            % assert(dest_len <= source_len)
    
            ac_slice(dest_start_i:dest_end_i) = ac_slice(dest_start_i:dest_end_i) * source_scale;
            
        end
    
    end
    autocorr(ac_i, :) = ac_slice;
end

end