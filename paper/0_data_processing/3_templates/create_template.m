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


function [template, beat_corrs, beat_indexes] = create_template(beat_indexes, all_beats, min_corr)
% create a template using a given set of beats (provided by the indexes)

% min_corr is excluded when not looking to interatively adjust it
if nargin < 3
    min_corr = [];
end

beats = all_beats(beat_indexes, :);

% catch any transpose issues when there is only one beat
if size(beats, 2) == 1
    beats = beats';
end

% create a robust template using the median of the robust_zscored beats
% for normally distributed data, these will be nearly equivalant
template = median(robust_zscore(beats')', 1);

% get the beat correlations to the template
beat_corrs = corr(beats', template');

% if no min_corr was provided, then skip iteratively raising the min_corr
if ~isempty(min_corr)
    % must do this in steps, to prevent cutting everything off all at once
    % start at cutting off everything below 0
    % then do in steps 1/num_steps of the min_corr
    num_steps = 5;

    step = 0;
    while min(beat_corrs) < min_corr
        current_min_corr = min((step/num_steps) * min_corr, min_corr);
        step = step + 1;
        
        % remove those indexes that are below the minimum
        beat_indexes(beat_corrs < current_min_corr) = [];
        
        % this should not happen, but exit if it does
        if isempty(beat_indexes)
            break
        end
           
        % recompute
        [template, beat_corrs, beat_indexes] = create_template(beat_indexes, all_beats);
    end
end

end
