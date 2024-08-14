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


function [template, ecg_indexes, used_beats] = use_archetypes_to_find_template(archetype_templates, ecg, lv_peaks, template_width, template_center, min_corr)
% use archetypes to find template, and then iterate on that template


% get the list of all possible indexes
ecg_indexes = lv_peaks(:, 1);

% get the array of beats and filter the visible indexes
[beats, ecg_indexes] = get_array_of_beats(ecg, ecg_indexes, template_width, template_center);


% 1. find any beats that match the archetypes
[beat_indexes] = find_archetype_templates(archetype_templates, beats, min_corr);

% 2. create a record-specific template from those beats
[template, ~, beat_indexes] = create_template(beat_indexes, beats, min_corr);

% 3. look for other beats that match the record-specific template
[beat_indexes] = find_archetype_templates(template, beats, min_corr);

% 4. refine the template from those beats
[template, ~, beat_indexes] = create_template(beat_indexes, beats, min_corr);


% filter to just the list of indexes actually used
ecg_indexes = ecg_indexes(beat_indexes);


if nargout > 2
    % get the beats
    used_beats = beats(beat_indexes, :);
else
    used_beats = [];
end

end
