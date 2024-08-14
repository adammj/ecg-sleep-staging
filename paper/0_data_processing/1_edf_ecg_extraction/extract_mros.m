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
%clear
% load('/hdd_4tb/uh/2018/ecgs/cfs.mat')
input_directory = '/sleep_data/raw_datasets/nsrr/nchsdb/sleep_data/';
output_directory = '/sleep_data/raw_ecgs/mros/';

files = dir([input_directory, '**/*.edf']);



%%
input_directory = '/sleep_data/raw_datasets/nsrr/nchsdb/sleep_data/';
output_directory = '/sleep_data/raw_ecgs/nchsdb/';

count = length(files);
ppm = ParforProgressbar(count);
parfor i = 1:count
    %fprintf('file: %i/%i\n', file_i, size(edf_list,1));
    
    %signal_names = edf_list(file_i).ecg_only;
    filename = [files(i).folder, '/', files(i).name];
    outputfile = [output_directory, strrep(files(i).name, '.edf', '.mat')];

    if exist(outputfile, 'file')
        ppm.increment();
        continue
    end

    hdr = edfread(filename);
    
    output = struct();

    for j = 1:length(hdr.label)
        if ~isempty(strfind(upper(hdr.label{j}), 'ECG')) ||  ~isempty(strfind(upper(hdr.label{j}), 'EKG'))
            label = hdr.label{j};

            output.(label) = struct();
            output.(label).channel = label;
            
            [~, data] = edfread(filename, 'targetSignals', label);
            output.(label).fs = length(data)/hdr.records;
            output.(label).data = single(data(:));
        end
    end
    
    output.start_time = strrep(hdr.starttime, '.', ':');
    
    save_struct(outputfile, output, true, true);

    ppm.increment();

    % break
    % %signal_names = signal_names(:,1);
    % 
    % % visit_dir = 'visit1/';
    % % if ~isempty(strfind(edf_list(file_i).name, 'visit2'))
    % %     visit_dir = 'visit2/';
    % % end
    % 
    % %file_name = cat(2, input_directory, visit_dir, edf_list(file_i).name);
    % %input_directory = edf_list(file_i).folder;
    % %input_directory = [strrep(input_directory, '4tb', '10tb'), '/'];
    % %file_name = cat(2, input_directory, edf_list(file_i).name);
    % 
    % 
    % 
    % if ~isempty(signal_names)
    %     for channel_i = 1:length(signal_names)
    %         %fprintf('channel: %i/%i\n',channel_i, length(signal_names));
    % 
    %         channel = edf_list(file_i).ecg_only{channel_i}; %.label;
    %         if ~isempty(strfind(channel, 'cs_'))
    %             continue
    %         end
    % 
    %         data.(channel).channel = channel;
    % 
    %         [hdr, temp_data] = edfread(file_name, 'targetSignals', signal_names(channel_i));
    % 
    %         data.(channel).fs = hdr.samples;
    % 
    %         if isvector(temp_data)
    %             data.(channel).data = single(temp_data(:));
    %         else
    %             %fprintf('error with channel %i for file %i\n', channel_i, file_i);
    %         end
    %     end
    % else
    %     %fprintf('no signals ecg found\n');
    % end
    % 
    % outfile = cat(2,output_directory, strrep(edf_list(file_i).name, 'edf', 'mat'));
    % save_struct(outfile, data, true, true);
    
    % ppm.increment();
end

delete(ppm)
disp('done')
