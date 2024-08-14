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


function [prediction_indexes, internal_values] = select_lv_peaks(lv_peaks, length_of_signal, fs)

% default outputs
internal_values = [0,0,0];
prediction_indexes = [];

% store a backup to skip any step
lv_peaks_bak = lv_peaks;


%% 1) filter peaks by their width
% In a handful of very strange morphologies, the r-wave local variation peak
% width is larger than normal. However, in 95+% of the cases, the range is
% between 75 and 100ms.

% first pass, remove all of too-wide peaks
lv_peaks(lv_peaks(:,3) > (fs*0.14), :) = [];  %140ms

% now, get the histcounts of the remaining widths (in bins sized by samples)
wide_counts = histcounts(lv_peaks(:,3), 1:1:round(fs*0.2), "Normalization", "probability");

% find the peaks and sort in decending order
[wide_pks, wide_locs] = findpeaks(wide_counts);
[~, I] = sort(wide_pks, 'descend');
wide_locs = wide_locs(I);
wide_pks = wide_pks(I);

% if more than 1 peak, and the 2nd is both narrower and at least 50% of the
% probability of the tallest, select it instead
if length(I) > 1
    if (wide_pks(2) >= 0.5*wide_pks(1)) & (wide_locs(2)<wide_locs(1))
        wide_pks(1) = [];
        wide_locs(1) = [];
    end
end

% calculate the cutoff width by choosing the peak and adding 10%
wide_cutoff = wide_locs(1)*1.1;

% and eliminate those above the cutoff
lv_peaks(lv_peaks(:, 3) > wide_cutoff, :) = [];

fprintf('after wide: %i\n', max(lv_peaks(:,1)));

%lv_peaks = lv_peaks_bak;


%% 2) filter peaks by their noise levels
% slowly adjust the threshold to get a desired value

edge_div = 1;
edges = 0:edge_div:16;
sec_of_signal = round(length_of_signal/fs);
count_threshold = min(round(sec_of_signal/6), 2000);
noise_threshold = 3;
noise_max = 5;

while noise_threshold <= noise_max

    % only keep those with lower SNR
    filtered_lv_peaks = lv_peaks((lv_peaks(:, 4) <= noise_threshold), :);
    counts = histcounts(filtered_lv_peaks(:, 2), edges);
    
    if max(counts) > count_threshold
        break
    end
    noise_threshold = noise_threshold + 0.2;
end

internal_values(1) = noise_threshold;

% if the threshold was at or above the max, then cut the count in half
if noise_threshold >= noise_max
    count_threshold = count_threshold /2;
end

% get the counts at or above that threshold
height_locs = find(counts >= count_threshold);

fprintf('after noise: %i\n', max(filtered_lv_peaks(:,1)));

if ~isempty(height_locs)
    % set the peak height bounds
    % around 65% some tall t-waves start getting selected
    height_low = (edge_div*height_locs(end))*0.65;  
    height_high = (edge_div*height_locs(end))*1.35;

    internal_values(2) = height_low;
    internal_values(3) = height_high;
    
    % remove those below the low height limit
    filtered_lv_peaks(filtered_lv_peaks(:, 2) < height_low, :) = [];
    % disp(length(filtered_lv_peaks))
    %filtered_lv_peaks(filtered_lv_peaks(:, 2) > height_high, :) = [];  %bad idea?

    % set the peak ac_wide bounds
    ac_div = 5;
    max_div = round(0.6*fs/ac_div)*ac_div;
    ac_counts = histcounts(lv_peaks(:, 6), 0:ac_div:max_div, 'Normalization', 'probability');
    
    % find the peaks and sort
    [ac_pks, ac_locs] = findpeaks(ac_counts);
    [~, I] = sort(ac_pks, 'descend');
    ac_locs = ac_div*ac_locs(I);

    if ~isempty(ac_locs)
        if ac_locs(1) <= (0.25*fs)
            filtered_lv_peaks(filtered_lv_peaks(:, 6) >= (0.3*fs), :) = [];
        end
    end

    locs = filtered_lv_peaks(:, 1);
    [~, org_I] = ismember(locs, lv_peaks(:, 1));

    prediction_indexes = org_I;

else
    fprintf('no peaks found in height\n')
end


% done!




%% 1) select a percent of the tallest peaks
% steady median kappa increase until 50%, and then decline and false positives take off
% 30% looks like a good compromise of kappa and fp
% some of these will be false positives
% percent = 0.3;
% number_of_peaks = round(percent*size(lv_peaks(:,1),1));
% 
% [~, I] = topkrows(lv_peaks(:,1), number_of_peaks);
% prediction_indexes = sort(I);
% 
% return


%% doing new code here


% height_cutoff = 3;
% per_cutoff = 0.5;
% 
% desired_count = round(per_cutoff*size(lv_peaks,1));
% sec_of_signal = round(length_of_signal/fs);
% desired_count = min(desired_count, sec_of_signal);
% 
% lv_peaks_min_height = lv_peaks(lv_peaks(:,2) >= height_cutoff, :);
% 
% [~, top_heights] = topkrows(lv_peaks_min_height(:,2), desired_count);
% [~, bottom_noise] = topkrows(lv_peaks_min_height(:,4), desired_count, 'ascend');
%  
% I = top_heights(ismember(top_heights, bottom_noise));
% % fprintf('%i %i\n', length(top_heights), length(I));
% if length(I) < (desired_count/4)
%     fprintf('reset\n');
%     I = top_heights;
% end
% sub_heights = lv_peaks_min_height(I, 2);
% sub_noises = lv_peaks_min_height(I, 4);
% med_sub_heights = median(sub_heights);
% med_sub_noises = median(sub_noises);
% max_noise = max(3, med_sub_noises * 1.2);
% max_height = med_sub_heights * 1.35;
% min_height = max(height_cutoff, med_sub_heights * 0.65);
% 
% filtered_lv_peaks = lv_peaks;
% % filtered_lv_peaks(filtered_lv_peaks(:,2) > max_height, :) = [];
% filtered_lv_peaks(filtered_lv_peaks(:,2) < min_height, :) = [];
% filtered_lv_peaks(filtered_lv_peaks(:,4) > max_noise, :) = [];
% 
% [~, org_I] = ismember(filtered_lv_peaks(:,1), lv_peaks(:, 1));
% prediction_indexes = org_I;


%% attempt 3
% distance = sqrt((0-lv_peaks(:, 9)).^2 + (16-lv_peaks(:, 2)).^2);
% 
% 
% [~, I] = topkrows(distance, min(20000, sec_of_signal), 'ascend');
% 
% % locs = lv_peaks(I, 1);
% % [~, org_I] = ismember(locs, lv_peaks(:, 1));
% 
% prediction_indexes = I;

end