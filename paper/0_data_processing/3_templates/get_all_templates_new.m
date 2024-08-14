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
template_width = 1*fs;
template_center = 100;
min_corr = 0.5;
total_samples_per_slice = 2*(2*fs);
constants = get_constants(fs);
constants.autocorr.bounds(1) = 1;  % ignore 0 lag

count = height(files_df);
%ppm = ProgressBar(count, 'something', 'cli');

ecg_folder = '/sleep_data/resampled_ecgs/';
% peaks_folder = '/sleep_data/resampled_lv_peaks/';

tic
parfor i = 1:count 
    %fprintf('%i\n', i)
    filename_ecg = [ecg_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    % filename_lv = [peaks_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    
    % if files_df(i, :).template_count > 0
    %     % disp('here')
    %     ppm.increment();
    %     continue;
    % end

    % try
        data_ecg = load(filename_ecg);  %load all of the data
        % data_lv = load(filename_lv, 'lv_peaks');
        
        % if lv_peaks isn't there, then create it
        if ~isfield(data_ecg, 'lv_peaks')
            % calculate lv and peaks
            lv = calculate_local_variation(data_ecg.ecg, constants.local_variation.local_samples, constants.local_variation.group_samples);
            lv = filter_local_variation(lv, constants.local_variation.lp_filter);
    
            [pks, locs, w] = findpeaks(lv, 'MinPeakDistance', round(0.3*fs), 'MinPeakHeight', 1.5, 'WidthReference', 'halfheight');
            if ~isempty(pks)
                lv_peaks = [locs, pks, w];
            else
                lv_peaks = [];
            end
            lv_peaks(:, 4) = snr_lv(lv, fs, lv_peaks);

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

            data_ecg.lv_peaks = lv_peaks;
            
        end
    
        % get the template and locations
        [template, template_locations] = use_archetypes_to_find_template(archetype_templates, data_ecg.ecg, data_ecg.lv_peaks, template_width, template_center, min_corr);
        
    
        % get counts for each epoch
        epoch_template_counts = zeros(length(data_ecg.ecg), 1, 'single');
        epoch_template_counts(template_locations) = 1;
        epoch_template_counts = reshape(epoch_template_counts', 30*fs, []);
        epoch_template_counts = sum(epoch_template_counts, 1)';
        epochs_with_a_template_percent = length(epoch_template_counts(epoch_template_counts>0))/length(epoch_template_counts)*100;
    
    
        % store data
        % files_df(i, :).template_count = length(template_locations);
        % files_df(i, :).template_count_div_epochs = files_df(i, :).template_count / data_ecg.epochs;
        % files_df(i, :).epochs_with_a_template_percent = epochs_with_a_template_percent;
        
        % update the structure and save
        data_ecg.template = template;
        data_ecg.template_locations = template_locations;
        data_ecg.epoch_template_counts = epoch_template_counts;
        data_ecg.epochs_with_a_template_percent = epochs_with_a_template_percent;
        
        %disp('would save here')
        save_struct(filename_ecg, data_ecg, true, true);
    % catch
    % end

    % count(ppm);
end
toc
disp('done')

% delete(ppm)



%%
% parfor i = 1:2884
% 
%     filename = [files(i).folder, '/', files(i).name];
%     filename_lv = strrep(filename, 'resampled_ecgs', 'resampled_lvs');
% 
%     data = load(filename, 'ecg');
% 
%     files(i).epoch_mad = mad(reshape(data.ecg, (30*256), []), 1)';
% end
% 
% %%
% 
% for i = 1:2884
%     files(i).epoch_mad_min = min(files(i).epoch_mad);
% end
