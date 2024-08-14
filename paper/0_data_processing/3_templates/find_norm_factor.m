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
clc
fs = 256;
template_width = 1*fs;
template_center = 100;
min_corr = 0.5;

% count_files = 4000;
count_files = height(files_df);
ppm = ProgressBar(count_files);

ecg_folder = '/sleep_data/resampled_ecgs/';
% peaks_folder = '/sleep_data/resampled_lv_peaks/';
% ecg_folder = '/dataset/';


% first calculate, later store
tic
parfor i = 1:count_files

    filename_ecg = [ecg_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    % filename_ecg = [files(i).folder, '/', files(i).name];
    % filename_lv = [peaks_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    
    %try
        data_ecg = load(filename_ecg, 'ecg', 'template', 'template_locations'); %, 'beat_probabilities', 'normalization_factor');  % load all of the data
        %data_lv = load(filename_lv, 'lv_peaks');
    
        % only have to do this again, because I erased the locations
        %[template, locations, beats] = use_archetypes_to_find_template(archetype_templates, data_ecg.ecg, data_lv.lv_peaks, template_width, template_center, min_corr);
        
        % they do exist
        template = data_ecg.template;
        locations = data_ecg.template_locations;
        
        % take the locations that have an 80% probability or greater
        % locations = data_ecg.beat_probabilities(data_ecg.beat_probabilities(:,2)>=0.8, 1);
        
        % take all of the locations used
        % locations = [data_ecg.beats_real; data_ecg.beats_fake];
        
        [beats, ~] = get_array_of_beats(data_ecg.ecg, locations, template_width, template_center);
        assert(size(beats, 1) == length(locations))
    
        
        % only if it already matches the length, which it should
        %if abs(length(locations) - files_df(i, :).template_count) <= 1
    
            beatsabs = abs(beats);
            beatsmax = max(beatsabs, [], 2);
            prctiles = prctile(beatsmax, 50:5:100);
            
            normalization_factor = prctiles(end-2) * 2;
            files_df(i, :).normalization_factor = normalization_factor;
            files_df(i, :).over_percent = sum((abs(data_ecg.ecg / normalization_factor) >= 1))/length(data_ecg.ecg)*100;
            
            % files(i).norm_factor_old = data_ecg.normalization_factor;
            % files(i).normalization_factor = normalization_factor;
            % files(i).over_percent = sum((abs(data_ecg.ecg / normalization_factor) >= 1))/length(data_ecg.ecg)*100;
        % else
        %     files_df(i, :).normalization_factor  = -1;
        %     files_df(i, :).over_percent = -1;
        % end
    % catch
    % end

    count(ppm);
end
toc

delete(ppm)
% clear ppm




%%

% store the variables in the ecg files
disp('store factor')
for i = 1:height(files_df)
    filename_ecg = [ecg_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    normalization_factor = files_df(i, :).normalization_factor;
    over_percent = files_df(i, :).over_percent;
    save(filename_ecg, "normalization_factor", "over_percent", "-append");
end

disp('done')



%%

% beatsabs = abs(beats);

% plot(corr(beats', template'), max(beatsabs, [], 2), '.');

% find the ratio wrt to the 90th prctile (to see if 2x that would likely
% capture everything)
% for i = 1:1000
%     plot(files(i).prctiles(1:(end-1))./files(i).prctiles(end-2))
%     hold on
% end
% hold off
% ylim([0, 4])

%%
% prctiles = zeros(9562, 11);
% for i = 1:9562
%     prctiles(i, :) = files(i).prctiles./files(i).prctiles(end-2);
% end

%%

% plot(median(prctiles))
% hold on
% plot(min(prctiles))
% plot(max(prctiles))
% hold off
% ylim([0, 4])


%%

% for factor = 20:20:200
% 
%     possiblei = find(([files.norm_factor] >= (factor-5)) & ([files.norm_factor] <= (factor+5)));
% 
%     for j = 1:2
%         i = randsample(possiblei, 1);
%         fprintf('%i  %i\n', factor, i);
%         filename = [files(i).folder, '/', files(i).name];
%         filename = strrep(filename, 'raw_ecgs', 'resampled_ecgs');
%         data_ecg = load(filename, 'ecg');
% 
%         plot(data_ecg.ecg./files(i).norm_factor);
%         ylim([-1, 1])
%         pause
%     end
% end
% 
% 
% 
% %%
% for i = 1:736
% 
%     files(i).arch1 = abs(corr(archetype_templates(1, :)', files(i).template'));
%     files(i).arch2 = abs(corr(archetype_templates(2, :)', files(i).template'));
%     files(i).arch3 = abs(corr(archetype_templates(3, :)', files(i).template'));
%     files(i).arch4 = abs(corr(archetype_templates(4, :)', files(i).template'));
% end
        

