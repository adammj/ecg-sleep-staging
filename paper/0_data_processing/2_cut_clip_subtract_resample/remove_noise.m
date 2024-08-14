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


function [signal] = remove_noise(signal, fs, filter_bw, noise_freqs)
%Remove all individual noisy frequencies, above the low-pass frequency

assert(isvector(signal))
    
if isempty(noise_freqs)
    return
end

%remove each noise frequency given
for i = 1:length(noise_freqs)
    [b, a] = iirnotch(noise_freqs(i)/(fs/2), filter_bw/(fs/2));
    signal = filtfilt(b, a, signal);
end

end
