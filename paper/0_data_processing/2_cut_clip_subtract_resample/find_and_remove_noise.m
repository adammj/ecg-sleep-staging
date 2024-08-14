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


function [ecg, noise_freqs] = find_and_remove_noise(ecg, fs, ecg_min_freq, ecg_noise_min_ratio, ecg_filter_bw, max_freqs)

% remove other noises above a minimum frequency
noise_freqs = [];

%find and remove each noise frequency, in descending order
while 1
    temp_noise_freqs = find_noise(ecg, fs, ecg_min_freq, ecg_noise_min_ratio);
    
    %remove freqs already removed
    if ~isempty(noise_freqs) && ~isempty(temp_noise_freqs)
        i = length(temp_noise_freqs);
        while i > 0
            if isempty(temp_noise_freqs)
                break;
            end
            
            for j = 1:length(noise_freqs)
                if isempty(temp_noise_freqs)
                    break;
                end

                if temp_noise_freqs(i) == noise_freqs(j)
                    temp_noise_freqs(i) = [];
                    break; %have to exit the inner loop, now that one has been removed
                end
            end
            
            i = i - 1;
        end
    end
    
    %if no noise frequencies, then exit loop
    if isempty(temp_noise_freqs)
        break;
    end

    %take only the first (and largest) frequency
    temp_noise_freqs = temp_noise_freqs(1);

    %append to the list
    noise_freqs = cat(1, noise_freqs, temp_noise_freqs);

    %filter from the ecg
    ecg = remove_noise(ecg, fs, ecg_filter_bw, temp_noise_freqs);
    
    % if the max count was reached, stop
    if length(noise_freqs) >= max_freqs
        break;
    end
end