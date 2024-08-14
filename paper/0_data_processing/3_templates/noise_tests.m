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




%%
clear
clc

folders = {'ccshs','cfs','chat_baseline','chat_followup','chat_nonrandomized'};
files = struct([]);
for i = 1:length(folders)
    files_part = dir(['/raw_ecgs/', folders{i}, '/*.mat']);
    files = [files; files_part];
end
% remove extraneous variables
clear files_part folders



%%
windowsec = 10;  %to get nicer fdivs

count = length(files);
ppm = ParforProgressbar(count, 'showWorkerProgress', true);

for i = 1219 %1:count
    filename = [files(i).folder, '/', files(i).name];
    filename = strrep(filename, 'raw_', 'resampled_');
    data = load(filename);
    field_names = fields(data);
    ecg = data.(field_names{1}).data;
    fs = data.(field_names{1}).fs;


    % reshape
    full_count = floor(length(ecg)/(windowsec*fs))*(windowsec*fs);
    ecg = ecg(1:full_count);

    ecg_shaped = reshape(ecg, (windowsec*fs), [])';
    
    % get the mad and then sort by distance from the median
    epoch_mad = mad(ecg_shaped', 1)';
    epoch_mad = abs(epoch_mad - median(epoch_mad));
    
    [~, sorted_indx] = sort(epoch_mad);
    
    % take the 3 epochs nearest the median mad
    
    L = windowsec*fs;
    f = fs*(0:(L/2))/L;
    
    counttoavg = 10;
    output = zeros(counttoavg, (L/2+1));
    output_scaled = zeros(counttoavg, (fs/2+1));
    
    for j = 1:counttoavg
        Y = fft(ecg_shaped(sorted_indx(j), :));
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        output(j, :) = P1;
        output_scaled(j, :) = scaled_fft(ecg_shaped(sorted_indx(j), :), fs, 1);
    end

    files(i).f = f;
    files(i).median_out = median(output);
    files(i).scaled_fft = median(output_scaled);

    ppm.increment();
end

delete(ppm)
disp('done')
close all
%plot(f, median(output))

%%

for i= 1:2884
    files(i).first = files(i).f(2);
    files(i).len = length(files(i).median_out);
end
%%

all_fft = zeros(2884, 641);
all_fft_s = zeros(2884, 65);
for i = 1:2884
    all_fft(i, :) = files(i).median_out(1:641)/sum(files(i).median_out(1:641));
    all_fft_s(i,:) = files(i).scaled_fft(1:65);


    files(i).noise = find( all_fft_s(i,2:end) > 5);
end



