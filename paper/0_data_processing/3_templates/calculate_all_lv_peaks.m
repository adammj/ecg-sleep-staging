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

% constants that don't change
fs = 256;
total_samples_per_slice = 2*(2*fs);
constants = get_constants(fs);
constants.autocorr.bounds(1) = 1;  % ignore 0 lag
min_distance = 32;

% file_count = height(files_df);
file_count = 4000;
ppm = ProgressBar(file_count);

% input_folder = '/sleep_data/resampled_ecgs/';
% output_folder = '/sleep_data/resampled_lv_peaks/';

tic
parfor i = 1:file_count 
    
    %filename = [input_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    % filename_lv = [output_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    filename = [files(i).folder,'/',files(i).name];
    files(i).lv_peak_count = 0;
    
    %FIXME skip if output already exists
    % if exist(filename_lv, 'file')
    %     ppm.increment()
    %     continue
    % end
    
    try
        data = load(filename, 'lv'); %'ecg', 
    
        % calculate lv and peaks
        % lv = calculate_local_variation(data.ecg, constants.local_variation.local_samples, constants.local_variation.group_samples);
        % lv = filter_local_variation(lv, constants.local_variation.lp_filter);
    
        [pks, locs, w] = findpeaks(data.lv, 'MinPeakDistance', min_distance-1, 'MinPeakHeight', constants.local_variation.min_peak_height, 'WidthReference', 'halfheight');
        if ~isempty(pks)
            lv_peaks = [locs, pks, w];
        else
            lv_peaks = [];
        end
        lv_peaks(:, 4) = snr_lv(data.lv, fs, lv_peaks);
    
        if false
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
        end
        
        % save lv stuff for now
        output = struct();
        %output.lv = lv;  % don't save the lv, too much space
        output.lv_peaks = lv_peaks;
        % save_struct(filename_lv, output, true, true);
        save_struct(filename, output, true, true, true);

        % files_df(i, :).lv_peak_count = size(lv_peaks, 1);
        files(i).lv_peak_count = size(lv_peaks, 1);
    catch
    end

    count(ppm);
end
toc

delete(ppm)
disp('done')
