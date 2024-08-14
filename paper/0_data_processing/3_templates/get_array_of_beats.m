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


function [beats, ecg_indexes] = get_array_of_beats(ecg, ecg_indexes, template_width, template_center)
% get the array of beats and also filter out the indexes that can't be
% shown

% remove the indexes that can't have their full shape shown
ecg_indexes(ecg_indexes < (template_center)) = [];
ecg_indexes(ecg_indexes > (length(ecg) - (template_width - template_center))) = [];

% create the beats
beat_locs = (ecg_indexes-template_center)*ones(1, template_width);
beat_locs = beat_locs + ones(length(ecg_indexes), 1)*(1:template_width);
beats = ecg(beat_locs);

end