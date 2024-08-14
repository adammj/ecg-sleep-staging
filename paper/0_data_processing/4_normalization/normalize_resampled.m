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


%% this is just to normalize all of the already resampled records

% normalize using robust_zscore / 50
% and clip everything outside +/- 1
% this seems to be an adequate compromise on removing the huge
% voltage outliers, without affecting -most- normal heartbeats
% (for most recordings they are untouched, but when the heartbeat
%  voltage is skewed completely to one side or the other, then it can clip
%  the r-wave just slightly)

clear
clc

folders = {'ccshs', 'cfs', 'chat', 'mesa', 'wsc'};
files = struct([]);
for i = 1:length(folders)
    files_part = dir(['/sleep_data/resampled_ecgs/', folders{i}, '/*.mat']);
    files = [files; files_part];
end
clear files_part


%%

limit = 1;
zscore_div = 50;

count = length(files);
ppm = ParforProgressbar(count);

parfor i = 1:count
    filename = [files(i).folder, '/', files(i).name];
    data = load(filename);
    
    % normalize
    ecg_new = robust_zscore(data.ecg)/zscore_div;
    ecg_new(ecg_new > limit) = limit;
    ecg_new(ecg_new < -limit) = -limit;
    
    % store
    data.ecg = ecg_new;
    
    % calculate total percent
    outliers = (abs(ecg_new) >= limit);
    data.outlier_percent = sum(outliers)/length(outliers)*100;
    
    % get the epoch-specific percents
    % first, outlier
    outliers = reshape(outliers, (data.fs*30), [])';
    data.epoch_outlier_percent = single(sum(outliers,2)/(data.fs*30)*100);

    % then, clipped (which wasn't done before)
    if ~isempty(data.mask_array)
        clipped = reshape(data.mask_array, (data.fs_original*30), [])';
        clipped = 1 - clipped;  % invert mask array
        data.epoch_clipped_percent = single(sum(clipped,2)/(data.fs_original*30)*100);
    else
        data.epoch_clipped_percent = zeros(size(data.epoch_outlier_percent), 'single');
    end
    
    %fprintf('%i: %.4f\n', i, data.outlier_percent);

    save_struct(filename, data, true, true);
    ppm.increment();
end

disp('done')
delete(ppm)