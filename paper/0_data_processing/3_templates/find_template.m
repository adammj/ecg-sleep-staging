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


function [final_template, lv_indexes_used] = find_template(lv_peaks, predicted_lv_indexes, ecg, width, center, second_pass)

%trying to pick the area proximate to the r wave
% 15/30 778 and 0.731
% 20/40 743 and 0.731
% 30/60 685 and 0.731

assert(~isempty(predicted_lv_indexes))

if nargin < 6
    % disp('not enough')
    second_pass = false;
end
% disp(second_pass)

% start = 0; %30;
corr_threshold = 0.80;
additional_height_prctile = 25;


%% create the template from the prediction_indexes

% get the r_indexes
combined_ecg_indexes = lv_peaks(predicted_lv_indexes, 1);

% remove the indexes that can't have their full shape shown
combined_ecg_indexes(combined_ecg_indexes < (center)) = [];
combined_ecg_indexes(combined_ecg_indexes > (length(ecg) - (width - center))) = [];

% fprintf('before: %i\n', length(r_indexes));
% create the template, and remove indexes that correlate pooly with the majority
[temp_template, beat_corrs, combined_ecg_indexes] = create_template(combined_ecg_indexes, ecg, width, center, corr_threshold);
% fprintf('after: %i\n', length(r_indexes));

% store the template and indexes used in the initial pass
final_template = temp_template;
[~, initial_template_lv_indexes] = ismember(combined_ecg_indexes, lv_peaks(:, 1));


%% now, look at other peaks to consider
% ones that have a high lv height and correlate well with the template

peaks_to_consider = lv_peaks;
peaks_to_consider(initial_template_lv_indexes, :) = [];

% remove those below the limit of the peaks already selected
height_limit = prctile(lv_peaks(initial_template_lv_indexes, 2), additional_height_prctile);
peaks_to_consider(peaks_to_consider(:, 2) < height_limit, :) = [];

additional_ecg_indexes = peaks_to_consider(:, 1);

% remove the indexes that can't have their full shape shown
additional_ecg_indexes(additional_ecg_indexes < (center)) = [];
additional_ecg_indexes(additional_ecg_indexes > (length(ecg) - (width - center))) = [];

% disp(length(additional_ecg_indexes))

if second_pass & ~isempty(additional_ecg_indexes)  %FIXME skip this for now
    % disp('here')

    beat_locs = (additional_ecg_indexes-center)*ones(1, width);
    beat_locs = beat_locs + ones(length(additional_ecg_indexes), 1)*(1:width);
    beats = ecg(beat_locs);
    
    % get the beat correlations to the template
    if size(beats, 2) == 1
        beats = beats';
    end
    
    beat_corrs = corr(beats', temp_template');
    additional_ecg_indexes(beat_corrs < corr_threshold, :) = [];
    
    % if there are some beats, add them
    if size(additional_ecg_indexes, 1) > 0

        [~, additional_lv_indexes] = ismember(additional_ecg_indexes, lv_peaks(:, 1));
        combined_lv_indexes = unique([initial_template_lv_indexes; additional_lv_indexes]);

        % create a new template
        combined_ecg_indexes = lv_peaks(combined_lv_indexes, 1);

        % remove the indexes that can't have their full shape shown
        combined_ecg_indexes(combined_ecg_indexes < (center)) = [];
        combined_ecg_indexes(combined_ecg_indexes > (length(ecg) - (width - center))) = [];
        
        % make sure final template has high correlation with all indexes
        [temp_template, beat_corrs, combined_ecg_indexes] = create_template(combined_ecg_indexes, ecg, width, center, corr_threshold);

        [~, combined_lv_indexes] = ismember(combined_ecg_indexes, lv_peaks(:, 1));
        
        
        final_template = temp_template;
        lv_indexes_used = combined_lv_indexes;
    else
        final_template = temp_template;
        lv_indexes_used = initial_template_lv_indexes;
    end
else
    final_template = temp_template;
    lv_indexes_used = initial_template_lv_indexes;
end

%% third pass

% use the final template to look for any remaining peaks and build a new
% template
% disp('final pass')
% disp(length(lv_indexes_used))

ecg_indexes = lv_peaks(:, 1);
[ecg_indexes] = find_archetype_templates(final_template, ecg_indexes, ecg, width, center);
[final_template, ~, ecg_indexes_used] = create_template(ecg_indexes, ecg, width, center, corr_threshold);
[~, lv_indexes_used] = ismember(ecg_indexes_used, lv_peaks(:, 1));

% disp(length(lv_indexes_used))

end
