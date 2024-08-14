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


function [noise_freqs, freq_array, mean_fft, waterfall] = find_noise(input, fs, min_freq, noise_min_ratio, num_snapshots)
% Takes an input signal, and takes 10 ffts across the signal (unless a
% number is specified). It uses those 10 "snapshots" to figure out what the
% likely noisy frequencies are

if nargin < 5
    num_snapshots = 10;
end

window_len = floor(length(input)/num_snapshots);
waterfall = zeros(fs+1, num_snapshots);
freq_divs = 0.5;

%first, find the median amplitude
amp = zeros(num_snapshots,1);
for i = 1:num_snapshots
    start_i = (i-1)*window_len + 1;
    section = input(start_i:(start_i+20*fs));
    peaks = findpeaks(section, 'MinPeakDistance', 2*fs);
    troughs = findpeaks(-section, 'MinPeakDistance', 2*fs);
    amp(i) = median(peaks) + median(troughs);
end
amp = median(amp, 'omitnan');

%if below a reasonable minimum amplitude
if amp < 0.1
    amp = 0.1;
end

for i = 1:num_snapshots
    start_i = (i-1)*window_len + 1;
    end_i = i*window_len;
    [waterfall(:,i), freq_array] = scaled_fft(input(start_i:end_i), fs, freq_divs, amp/2);
end

%create the mean fft from the waterfall plot
% (mean seems to work a little better at detecting noise that only occurs for
%  a part of the input)
mean_fft = mean(waterfall, 2, 'omitnan');

% range = struct('x',[],'y',[],'z',[]);
% range.z = [0, noise_min_ratio*2];
% surf_plot(waterfall, range);
% pause
% plot(freq_array, mean_fft);
% ylim([0, noise_min_ratio*2])
% pause

%build an array
noise_freqs = freq_array;
noise_freqs(:,2) = mean_fft;

%remove inapplicable frequencies
noise_freqs(noise_freqs(:,1) < min_freq, :) = [];
noise_freqs(noise_freqs(:,2) < noise_min_ratio, :) = [];
noise_freqs(isnan(noise_freqs(:,2)), :) = [];
noise_freqs(isinf(noise_freqs(:,2)), :) = [];
noise_freqs(noise_freqs == freq_array(end), :) = [];

%sort by decending ratio
noise_freqs = sortrows(noise_freqs, -2);

%just take the frequencies
noise_freqs = noise_freqs(:,1);


end
