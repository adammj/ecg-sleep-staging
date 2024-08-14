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


clc
%clear

folders = {'chat'};
files = struct([]);
for i = 1:length(folders)
    files_part = dir(['/sleep_data/resampled_lvs/', folders{i}, '/*.mat']);
    files = [files; files_part];
end
% remove extraneous variables
clear files_part folders

%%
clc

% constants that don't change
fs = 256;
total_samples_per_slice = 2*(2*fs);
constants = get_constants(fs);
constants.autocorr.bounds(1) = 1;  % ignore 0 lag
template_width = 1*fs;
template_center = 100;
% PR interval is normally 120-200ms
% QT interval is normally 350-450ms
% setting the center at 100 (391ms) seems like a good compromise

count = length(files);
fprintf('total: %i\n', count);
% ppm = ParforProgressbar(count);

for i = 1 %1:count
    filename = [files(i).folder, '/', files(i).name];
    
    % if creating the lv array and peaks
    if 1
        filename_ecg = strrep(filename, '/8tb_drive/resampled_lvs', '/sleep_data/resampled_ecgs');
        data = load(filename_ecg, 'ecg');

        % calculate lv and peaks
        lv = calculate_local_variation(data.ecg, constants.local_variation.local_samples, constants.local_variation.group_samples);
        lv = filter_local_variation(lv, constants.local_variation.lp_filter);

        [pks, locs, w] = findpeaks(lv, 'MinPeakDistance', round(0.3*fs), 'MinPeakHeight', 1.5, 'WidthReference', 'halfheight');
        if ~isempty(pks)
            lv_peaks = [locs, pks, w];
        else
            lv_peaks = [];
        end
        lv_peaks(:, 4) = snr_lv(lv, fs, lv_peaks);

        % save lv stuff for now
        output = struct();
        output.lv = lv;
        output.lv_peaks = lv_peaks;
        save_struct(filename, output, true, true);
    end

    % if updating the lv_peaks
    if 1
        %load lv and peaks to update lv_peaks
        data_lv = load(filename);
        
        % if already the correct size, then skip
        if size(data_lv.lv_peaks, 2) == 6
            ppm.increment();
            continue;
        end

        lv = data_lv.lv;
        lv_peaks = data_lv.lv_peaks;

        % calculate the ac
        slices = floor(length(lv)/total_samples_per_slice);
        lv_sliced = reshape(lv(1:(total_samples_per_slice*slices)), total_samples_per_slice, [])';
        ac_data = struct();
        ac_data.ac = zeros(slices, 2*fs, 'single');
        for ac_i = 1:slices
            ac_data.ac(ac_i, :) = autocorr_merged(lv_sliced(ac_i, :), constants.autocorr.bounds);
        end

        % smooth out the array
        ac_data.ac = smooth_autocorr(ac_data.ac, constants.autocorr);
        % filter
        ac_data.ac = filter_autocorr_harmonics(ac_data.ac, constants.autocorr.bounds(2:3), constants.autocorr.harmonic_lag_plus_minus);
        % ac_wide
        ac_data.wide_ac = calculate_wide_autocorr(ac_data.ac, constants.autocorr);

        % calculate the additional lv_peak_columns
        current_i_list = ceil(lv_peaks(:,1)/(4*fs));
        current_i_list(current_i_list > slices) = slices;
        temp_vec = sum(ac_data.ac, 2);
        lv_peaks(:, 5) = temp_vec(current_i_list);
        temp_vec = sum(ac_data.wide_ac, 2);
        lv_peaks(:, 6) = temp_vec(current_i_list);
        
        % store the new lv_peaks
        data_lv.lv_peaks = lv_peaks;
        save_struct(filename, data_lv, true, true);
    end

    % ppm.increment();
end

disp('done')
% delete(ppm)
