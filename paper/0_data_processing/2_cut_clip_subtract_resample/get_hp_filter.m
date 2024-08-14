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


function [hp_filter] = get_hp_filter(fs)
persistent hp_filters

hp_filter = [];
if ~isempty(hp_filters)
    for i = 1:size(hp_filters, 1)
        if hp_filters(i, 1).fs == fs
            hp_filter = hp_filters(i).filter;
            break;
        end
    end
end

if isempty(hp_filter)
    hp_filter = designfilt('highpassiir', 'SampleRate', fs, ...
        'DesignMethod', 'cheby2', ...    
        'PassbandFrequency', 0.5, ...
        'PassbandRipple', 0.1, ...
        'StopbandFrequency', 0.25, ...
        'StopbandAttenuation', 60, ...
        'MatchExactly', 'passband');
    
    %append to end
    hp_filters(end+1, 1).fs = fs;
    hp_filters(end, 1).filter = hp_filter;
end

end

