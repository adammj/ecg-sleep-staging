% Copyright (C) 2024  Adam M. Jones
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
load('./chat_nonrandomized.mat')
input_directory = '/large/nsrr/chat/polysomnography/edfs/nonrandomized/';
output_directory = '/small/ecgs/chat_nonrandomized/';


disp('first find where files start')
for file_i = 1:size(ecg_list,1)
    file_name = cat(2, output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
    if ~exist(file_name)
        fprintf('file %i does not exist\n', file_i);
        break
    end
end

% if file_i > 1
%     file_i = file_i - 1;
% 	file_name = cat(2, output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
%     fprintf('delete file %i (just in case)\n', file_i);
%     delete(file_name)
%     start_file_i = file_i;
% else
%     start_file_i = 1;
% end

parfor file_i = 1:size(ecg_list,1)
    %fprintf('file: %i/%i\n', file_i, size(ecg_list,1));
    
    signal_names = struct2cell(ecg_list(file_i).signals)';
    signal_names = signal_names(:,1);
    file_name = cat(2, input_directory, ecg_list(file_i).file);
    output_file = cat(2,output_directory, strrep(ecg_list(file_i).file, 'edf', 'mat'));
    
    % if already created, skip
    if isfile(output_file)
        continue;
    end
    
    data = struct([]);

    if ~isempty(signal_names)
        for channel_i = 1:length(signal_names)
            fprintf('channel: %i/%i\n',channel_i, length(signal_names));
            data(channel_i,1).channel = ecg_list(file_i).signals(channel_i).label;
            data(channel_i,1).fs = ecg_list(file_i).signals(channel_i).fs;
            [~, temp_data] = edfread(file_name, 'targetSignals', signal_names(channel_i));

            if isvector(temp_data)
                data(channel_i,1).data = single(temp_data(:));
            else
                fprintf('error with channel %i for file %i\n', channel_i, file_i);
            end
        end
    else
        fprintf('no signals ecg found\n');
    end
    
    ecg_data = struct();
    ecg_data.data = data;
    
    save_struct(output_file, ecg_data, true, true);
end

fprintf('done\n');
