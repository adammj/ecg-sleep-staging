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
clear
load('/hdd_4tb/uh/2018/ecgs/ccshs.mat')
input_directory = '/hdd_4tb/nsrr/ccshs/polysomnography/edfs/';
output_directory = '/hdd_2tb/ecgs/ccshs/';

disp('first find where files start')
for file_i = 1:size(ecg_list,1)
    file_name = cat(2, output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
    if ~exist(file_name)
        fprintf('file %i does not exist\n', file_i);
        break
    end
end

if file_i > 1
    file_i = file_i - 1;
	file_name = cat(2, output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
    fprintf('delete file %i (just in case)\n', file_i);
    delete(file_name)
    start_file_i = file_i;
else
    start_file_i = 1;
end

for file_i = start_file_i:size(ecg_list,1)
    %fprintf('file: %i/%i\n', file_i, size(ecg_list,1));
    
    signal_names = struct2cell(ecg_list(file_i).signals)';
    signal_names = signal_names(:,1);
    file_name = cat(2, input_directory, ecg_list(file_i).file);

    ecgs = struct();
    ecgs.data = struct([]);

    if ~isempty(signal_names)
        for channel_i = 1:length(signal_names)
            %fprintf('channel: %i/%i\n',channel_i, length(signal_names));
            ecgs.data(channel_i,1).channel = ecg_list(file_i).signals(channel_i).label;
            ecgs.data(channel_i,1).fs = ecg_list(file_i).signals(channel_i).fs;
            [~, temp_data] = edfread(file_name, 'targetSignals', signal_names(channel_i));

            if isvector(temp_data)
                ecgs.data(channel_i,1).data = single(temp_data(:));
            else
                %fprintf('error with channel %i for file %i\n', channel_i, file_i);
            end
        end
    else
        %fprintf('no signals ecg found\n');
    end
    
    filename = cat(2,output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
    save_struct(filename, ecgs, true, true);
end

fprintf('done\n');
