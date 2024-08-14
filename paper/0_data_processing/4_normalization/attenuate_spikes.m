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




%% first get for 5 sec windows
temp_ecgs = reshape(ecgs, [200*5], [])';
range = max(temp_ecgs, [], 2) - min(temp_ecgs, [], 2);

median_5sec = prctile(range, 90); %median(range);
mad_5sec = mad(range, 1);


%% now, do this for 1 sec windows and get their z-scores
temp_ecgs = reshape(ecgs, [200*1], [])';
range = max(temp_ecgs, [], 2) - min(temp_ecgs, [], 2);

zscore_1sec = (range - median_5sec)/mad_5sec;

noisy_sec = find(zscore_1sec > 3);

per_noisy = length(noisy_sec)/size(temp_ecgs, 1);

%%



ecg_folder = '/sleep_data/dataset_1_files/';
all_files = dir([ecg_folder,'*.mat']);
file_count = size(all_files, 1);

stats = struct([]);
stats(file_count, 1).name = '';
stats(1, 1).mad_per = 0;
stats(1, 1).max_per = 0;
stats(1, 1).epoch_per = 0;
% stats(1, 1).per_noise = 0;
% stats(1, 1).per_noise2 = 0;

parfor i = 1:file_count
    filename = all_files(i).name;
    stats(i, 1).name = filename;
    ecg_data = load([ecg_folder, filename], 'ecgs');
    ecgs = ecg_data.ecgs(:);

    % temp_ecgs = reshape(ecgs, [200*5], [])';
    % range = max(temp_ecgs, [], 2) - min(temp_ecgs, [], 2);
    % 
    % median_5sec = prctile(range, 95); %median(range);
    % mad_5sec = mad(range, 1);
    % 
    % temp_ecgs = reshape(ecgs, [200*1], [])';
    % range = max(temp_ecgs, [], 2) - min(temp_ecgs, [], 2);
    % 
    % zscore_1sec = (range - median_5sec)/mad_5sec;
    % 
    % noisy_sec = find(zscore_1sec > 3);
    % 
    % per_noisy = length(noisy_sec)/size(temp_ecgs, 1);
    % 
    % per_noisy2 = length(find(mad(reshape(ecgs, [200*1], []), 1) > 4*mad(ecgs, 1))) / size(reshape(ecgs, [200*1], []),2);

    % stats(i, 1).median = median_5sec;
    % stats(i, 1).mad = mad_5sec;
    % stats(i, 1).per_noise = per_noisy;
    % stats(i, 1).per_noise2 = per_noisy2;

    [~, per_mad_masked, per_max_masked] = normalize_ecg(ecgs, 2, 200);
    stats(i, 1).mad_per = per_mad_masked;
    stats(i, 1).max_per = per_max_masked;
    
    % figure out how many epochs have at least one template beat in them
    location_counts = zeros(size(ecgs));
    location_counts(template_and_locations(i).locations) = 1;
    location_counts = reshape(location_counts, 6000, []);
    stats(i, 1).epoch_per = length(find(sum(location_counts) > 0)) / (length(ecgs)/6000);
end


%%
%clear
clc

files = dir('/sleep_data/dataset_1_files/*.mat');

limit = 1;
div = 0.01;
edges = [-limit:div:(limit+div)];


count = 1737;
%counts_all = zeros(count, (length(edges)-1));

%parfor i = 1:count
for i = 1100
    data = load([files(i).folder, '/', files(i).name], 'ecgs');
    ecgs = data.ecgs(:);
    
    % [ecgs2, per_mad_masked, per_max_masked, mad_indx, ~, mask_mad, mask_max] = normalize_ecg(ecgs, 2, 200);
    % fprintf('mad: %.2f   max: %.2f\n', per_mad_masked*100, per_max_masked*100);
    
    % ecgs = data.ecgs(:);
    ecgs2 = robust_zscore(ecgs)/40;
    
    ecgs2(ecgs2 > limit) = limit;
    ecgs2(ecgs2 < -limit) = -limit;
    
    counts = histcounts(ecgs2, edges, "Normalization", "probability");
    
    counts_all(i, :) = counts;
    
    %
    % figure
    % ax1 = subplot(2,1,1);
    % plot(ecgs)
    % hold on
    % % plot(mask_mad)
    % %plot((mad_indx-1)*(200*2)+1, ones(size(mad_indx)), '.')
    % hold off
    % 
    % ax2 = subplot(2,1,2);
    % plot(ecgs2)
    % 
    % linkaxes([ax1,ax2],'x');
end
disp('done')

%%
plot(edges(1:(end-1)), mean(counts_all(1:count,:)));
hold on
plot(edges(1:(end-1)), mean(counts_all(1:count,:).^(1/8)));
plot(edges(1:(end-1)), median(counts_all(1:count,:).^(1/8)));
hold off
ylim([0,1])

%%
subplot(2,1,1)
plot(ecgs)
subplot(2,1,2)
plot(ecgs2)

%%

plot(sort(per_noise_3_mad))
hold on
plot(sort(per_noise_4_mad))
hold off

%%
% this just needs to be calculated once (for each direction)
sec_to_use = 2;
spline_pp = spline([0, 0.5, 1],[0, 0, 0.5, 1, 0]);
clip_len = (200*sec_to_use)*0.5; %half of sec_to_use
clip_values = (1:-1/(clip_len-1):0)';
clip_values = ppval(spline_pp, clip_values);

% this changes each time
% both lower and upper can be different
upper=0.5;
lower=0.1;  
plot((clip_values*(upper-lower))+lower); 
ylim([0,1])

% this should start from the lowest values and go upwards
% a list of all transitions (for each dir) should be created beforehand
% and then apply the transitions from lowest to highest
% this will force the "higher" (less attentuated) epochs to be
% slightly lower when next to a lower epoch (attenutating them further, but
% not un-attentuating the "worse" epochs.





%%

for i = 1:3060
    filename = all_files(i).name;
    data.noise_freqs = load([ecg_folder, filename], 'noise_freqs');
    stats(i, 1).noise_freq_count = length(data.noise_freqs.noise_freqs);
    if length(data.noise_freqs.noise_freqs) == 1
        stats(i, 1).noise_freq = data.noise_freqs.noise_freqs(1);
    end
end


%%
% mask_right = find(mask_mad(1:(end-1)) < mask_mad(2:end));
% mask_left = find(mask_mad(2:end) < mask_mad(1:(end-1)));
clc
fs = 200;
sec_to_use = 2;

% mask_mad = [1, 0.6, 0.2, 0, 0.2, 0, 0.8, 0.6, 0.8, 1]';
% mask_mad = repelem(mask_mad, (fs*sec_to_use));

mask_test = smooth_mask_array_nonbinary(mask_mad, 200);


plot(mask_mad(1:end))
hold on
plot(mask_test(1:end))
hold off

