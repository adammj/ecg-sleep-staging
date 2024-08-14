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




% find the templates and store where they are in the ecg

% input:
% ecg

% output:
% template (array)
% template locations (array)


% need lv peaks, which needs lv


%% get list of all files to process
%clear
clc

folders = {'mesa', 'wsc',};
files = struct([]);
for i = 1:length(folders)
    files_part = dir(['/sleep_data/resampled_ecgs/', folders{i}, '/*.mat']);
    files = [files; files_part];
end
% remove extraneous variables
clear files_part folders


%% get template data
clc
close all

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
% ppm = ParforProgressbar(count, 'showWorkerProgress', true);

%1315, 1889, 2209, 2770 (excluded because low fs)

for i = 1 %:count 
    %fprintf('%i\n', i);
    filename = [files(i).folder, '/', files(i).name];
    filename_lv = strrep(filename, 'resampled_ecgs', 'resampled_lv_peaks');

    if 1% files(i).template_count < 5000
        % try
            data = load(filename, 'ecg');
            
            % try
            %     % calculate lv and peaks
            %     lv = calculate_local_variation(data.ecg, constants.local_variation.local_samples, constants.local_variation.group_samples);
            %     lv = filter_local_variation(lv, constants.local_variation.lp_filter);
            % 
            %     [pks, locs, w] = findpeaks(lv, 'MinPeakDistance', round(0.3*fs), 'MinPeakHeight', 1.5, 'WidthReference', 'halfheight');
            %     if ~isempty(pks)
            %         lv_peaks = [locs, pks, w];
            %     else
            %         lv_peaks = [];
            %     end
            %     lv_peaks(:, 4) = snr_lv(lv, fs, lv_peaks);
            % 
            %     % save lv stuff for now
            %     output = struct();
            %     output.lv = lv;
            %     output.lv_peaks = lv_peaks;
            %     save_struct(filename_lv, output, true, true);
            % catch
            % end
            % 
            % continue
            
            % load lv and peaks
            % data_lv = load(filename_lv);
            % lv = data_lv.lv;
            % lv_peaks = data_lv.lv_peaks;

            % % calculate the ac
            % slices = floor(length(lv)/total_samples_per_slice);
            % lv_sliced = reshape(lv(1:(total_samples_per_slice*slices)), total_samples_per_slice, [])';
            % ac_data = struct();
            % ac_data.ac = zeros(slices, 2*fs, 'single');
            % for ac_i = 1:slices
            %     ac_data.ac(ac_i, :) = autocorr_merged(lv_sliced(ac_i, :), constants.autocorr.bounds);
            % end
            % 
            % % smooth out the array
            % ac_data.ac = smooth_autocorr(ac_data.ac, constants.autocorr);
            % % filter
            % ac_data.ac = filter_autocorr_harmonics(ac_data.ac, constants.autocorr.bounds(2:3), constants.autocorr.harmonic_lag_plus_minus);
            % % ac_wide
            % ac_data.wide_ac = calculate_wide_autocorr(ac_data.ac, constants.autocorr);
            % 
            % % calculate the additional lv_peak_columns
            % current_i_list = ceil(lv_peaks(:,1)/(4*fs));
            % current_i_list(current_i_list > slices) = slices;
            % temp_vec = sum(ac_data.ac, 2);
            % lv_peaks(:, 5) = temp_vec(current_i_list);
            % temp_vec = sum(ac_data.wide_ac, 2);
            % lv_peaks(:, 6) = temp_vec(current_i_list);
            
            % load just lv peaks
            data_lv = load(filename_lv, 'lv_peaks');
            lv_peaks = data_lv.lv_peaks;
            
            % lv_peaks(lv_peaks(:,3) < (fs*0.075), :) = [];
            % lv_peaks(lv_peaks(:,3) > (fs*0.135), :) = [];
            % 
            %counts_temp = histcounts(lv_peaks(:,3), 1:1:50, "Normalization","probability");
            %plot(counts_temp)
            % 
            % wide_counts(i, :) = counts_temp(1:51);
            %ppm.increment();
            
            
            % continue
            % fprintf('before select lv: %i\n', max(lv_peaks(:, 1)))

            % select the most likely lv peaks
            % prediction_indexes = select_lv_peaks(lv_peaks, length(data.ecg), fs);
            
            % fprintf('after select lv: %i\n', max(lv_peaks(prediction_indexes, 1)))
            % pause
            % continue
            % prediction_indexes_org = prediction_indexes;

            % grab all ecg_indexes
            ecg_indexes = lv_peaks(:, 1);
            
            [ecg_indexes] = find_archetype_templates(archetype_templates, ecg_indexes, data.ecg, template_width, template_center);
            [~, predicted_lv_indexes] = ismember(ecg_indexes, lv_peaks(:, 1));

            % futher refine that while building a template
            [r_wave_template, lv_indexes] = find_template(lv_peaks, predicted_lv_indexes, data.ecg, template_width, template_center, true);

            % convert locations to locations in original ecg
            locations = lv_peaks(lv_indexes, 1);
            
            %FIXME: delete
            % [r_wave_template, ~, ecg_indexes] = create_template(ecg_indexes, data.ecg, template_width, template_center, 0.8);
            % [ecg_indexes] = find_archetype_templates(r_wave_template, ecg_indexes, data.ecg, template_width, template_center);
            % [r_wave_template, ~, locations] = create_template(ecg_indexes, data.ecg, template_width, template_center);

            % get counts for each epoch
            epoch_template_counts = zeros(length(data.ecg), 1, 'single');
            epoch_template_counts(locations) = 1;
            epoch_template_counts = reshape(epoch_template_counts', 30*fs, []);
            epoch_template_counts = sum(epoch_template_counts, 1)';
            
            % store data
            files(i).r_wave_template = r_wave_template;
            files(i).template_locs = locations;
            files(i).template_count = length(locations);
            files(i).epoch_template_counts = epoch_template_counts;
            files(i).epochs_with_template_percent = length(epoch_template_counts(epoch_template_counts>0))/length(epoch_template_counts)*100;
        % catch
        %     % if failed, flag row
        %     files(i).r_wave_template = [];
        %     files(i).epochs_with_template_percent = -1;
        %     files(i).template_count = -1;
        % end
    end

    % ppm.increment();
end

% delete(ppm)
disp('done')
close all

return

%%


subplot(2,1,1)
plot(data.ecg)
hold on
% locations_org = lv_peaks(prediction_indexes_org, 1);
plot(locations, 0*ones(size(locations)), '+', 'LineWidth', 2)
% plot(locations_org, 0*ones(size(locations_org)), '+', 'LineWidth', 2)
%plot(r_indexes, 0*ones(size(r_indexes)), '+', 'LineWidth', 2)
hold off
subplot(2,1,2)
plot(template)

%%
plot(data.ecg)
hold on
% plot(prediction_indexes_org, ones(size(prediction_indexes_org)), 'o')
plot(lv_peaks(:,1), lv_peaks(:,4)./max(lv_peaks(:,4)), '+')
% plot(lv_peaks(:,1), lv_peaks(:,2)/10, 'o')
hold off

%%
epoch_template_counts = zeros(length(data.ecg), 1, 'single');
epoch_template_counts(locations) = 1;

epoch_template_counts = reshape(epoch_template_counts', 30*fs, []);
epoch_template_counts = sum(epoch_template_counts, 1);

plot(epoch_template_counts)



%%

%ppm = ParforProgressbar(count, 'showWorkerProgress', true);
for i = 593
    filename = [files(i).folder, '/', files(i).name];
    %filename_lv = strrep(filename, 'resampled_ecgs', 'resampled_lvs');
    data = load(filename, 'ecg');

    counts = histcounts(abs(diff(data.ecg)), 0:(std(data.ecg)/1000):std(data.ecg), 'Normalization','probability');
    
    files(i).zero_diff_percent = counts(1)*100;

    plot(data.ecg)
%    ppm.increment();
end
%delete(ppm)




%%



