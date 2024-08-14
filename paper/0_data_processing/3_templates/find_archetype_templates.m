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


function [beat_indexes] = find_archetype_templates(archetype_templates, beats, min_corr)
% use archetype templates to find possible beats within a recording

assert(size(archetype_templates, 2) == size(beats, 2))

% get the correlations with each archetype
beat_indexes = [];
for archetype_i = 1:size(archetype_templates, 1)
    template = archetype_templates(archetype_i, :);

    % take the abs, making no assumption about the polarity
    beat_corrs = abs(corr(beats', template'));

    beat_indexes_for_archetype = find(beat_corrs >= min_corr);
    beat_indexes = [beat_indexes; beat_indexes_for_archetype];
end

% just take the unique set
beat_indexes = unique(beat_indexes);

end
