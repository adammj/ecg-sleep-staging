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



% working with raw ecg to take it through the resampled stage

% steps:
% 1) cut ecg to length (epoch count * 30 * fs) = floor(len/fs/30)*(fs*30)
% 2) mask out loose lead sections
%    a) for single channel: just remove sections and store mask
%    b) for two channels: remove sections from each, store single mask
% 3) for 2-channel, subtract ECG1-ECG2 (on random inspection this most often
%      produces the correct polarity, but not always)
% 4) remove baseline wander (<0.5 Hz)
% 5) remove 60 Hz with notch
% 6) remove additional noise with notch
% 7) resample to 256 Hz (if already 256, do nothing)
% 8) normalize (robust z-score, but don't clip)

% inputs: 
% 1) raw ecg file

% output:
% 1) resampled ecg file





%% process each file
clc

% constants (that don't depend on fs)
desired_fs = 256;
clip_ratio = 0.5;
ecg_filter_bw = 1.5;
ecg_min_freq = 10;
ecg_noise_min_ratio = 4;
max_additional_freqs = 10;

% precalculate some of the hp_filters
fs_options = [100, 125, 128, 200, 250, 256, 500, 512];
for fs = fs_options
    hp_filter = get_hp_filter(fs);
end

count = height(files_df);
% ppm = ParforProgressbar(count); 

input_folder = '/sleep_data/raw_ecgs/';
output_folder = '/sleep_data/resampled_ecgs/';


tic
parfor i = 1:count
    
    inputfilename = [input_folder, files_df(i,:).dataset{1}, '/', files_df(i,:).source_file{1}, '.mat'];
    outputfilename = [output_folder, files_df(i,:).dataset{1}, '/', files_df(i,:).source_file{1}, '.mat'];
    
    % skip the already created ones (FIXME: remove this)
    if exist(outputfilename, 'file')
    %     %disp('already')
        % ppm.increment();
        continue
    end

    is_stages_dataset = strcmp(files_df(i,:).dataset{1}, 'stages');
    
    % load data
    data = load(inputfilename);

    % prepare the output
    output = struct();

    % copy over the start time
    %if ~is_stages_dataset
        output.start_time = data.start_time;
    % if is_stages_dataset
    %     output.start_time = strrep(files_df(i,:).start_time{1}, '.', ':');
    % end
    
    % prepare for all possibilties seen
    channel_1 = [];
    channel_2 = [];

    % original code for matching the correct channels is kept at the end
    % the new code assumes that the files struct contains a channels_used
    % column that can be read
    
    % save struct and move to next if no channels are to be used
    if strcmp(files_df(i,:).channels_used{1}, 'NONE') | strcmp(files_df(i,:).channels_used{1}, 'UNKNOWN')
        output.channels_used = files_df(i,:).channels_used{1};
        save_struct(outputfilename, output, true, true);
        % ppm.increment();
        continue
    end

    % split by subtract sign
    channels_used_list = strsplit(files_df(i,:).channels_used{1}, '-');
    assert(length(channels_used_list) <= 2)
    
    channel_1 = data.(channels_used_list{1}).data;    
    fs = data.(channels_used_list{1}).fs;

    if length(channels_used_list) == 2
        channel_2 = data.(channels_used_list{2}).data;
        assert(length(channel_1) == length(channel_2))
    end
    
    % constants that depend on fs
    clip_length = next_odd(0.3*fs);
    fill_gap_length = round(1.0*fs);
    smooth_transition_length = round(1.0*fs);
    
    % store
    output.channels_used = files_df(i,:).channels_used{1}; 
    output.fs_original = fs;

    % cutting procedure for stages dataset
    if is_stages_dataset
        % cut off the sec from the start
        samples_to_cut = fs*files_df(i,:).sec_to_cut_from_start;
        channel_1(1:samples_to_cut) = [];
        
        % trim to the correct length
        samples_to_keep = fs*files_df(i,:).sec_to_keep;
        channel_1 = channel_1(1:samples_to_keep);
        
        % do the same for channel_2
        if ~isempty(channel_2)
            channel_2(1:samples_to_cut) = [];
            channel_2 = channel_2(1:samples_to_keep);
        end
    end


    % get correct length
    epochs = floor(length(channel_1)/fs/30);
    output.epochs = epochs;
    correct_length = epochs*(fs*30);

    % cut to correct length
    channel_1 = channel_1(1:correct_length);
    if ~isempty(channel_2)
        channel_2 = channel_2(1:correct_length);
    end
    
    %
    % mask out loose leads
    %
    if isempty(channel_2)
        % if only one channel, just apply mask that is calculated
        [channel_1, ~, output.mask_array] = remove_large_clips(channel_1, clip_length, clip_ratio, fill_gap_length, smooth_transition_length);
    
    else

        % get mask for channel 1
        [~, ~, mask_1_array] = remove_large_clips(channel_1, clip_length, clip_ratio, fill_gap_length, smooth_transition_length);
    
        % get mask for channel 2
        [~, ~, mask_2_array] = remove_large_clips(channel_2, clip_length, clip_ratio, fill_gap_length, smooth_transition_length);
        
        % get the combined mask by taking the minimum of the two masks
        % this is to make sure that whenever one channel is clipped out
        % the other is also removed, to prevent issues with information
        % leaking in that would've been subtracted out
        combined_mask_array = min(mask_1_array, mask_2_array);
        output.mask_array = combined_mask_array;

        % now, apply the same mask to both channels
        % channel 1
        median_value = median(channel_1.*combined_mask_array, 'omitnan');
        channel_1 = (channel_1 - median_value).*combined_mask_array;

        % channel 2
        median_value = median(channel_2.*combined_mask_array, 'omitnan');
        channel_2 = (channel_2 - median_value).*combined_mask_array;
    end
    
    % get the clipped ratio
    output.clipped_percent = (1 - sum(output.mask_array)./length(output.mask_array))*100;

    % get the epoch-specific clipped percent
    clipped = reshape(output.mask_array, (output.fs_original*30), [])';
    clipped = 1 - clipped;  % invert mask array
    output.epoch_clipped_percent = single(sum(clipped,2)/(output.fs_original*30)*100);
    
    % if clipped_percent = 0, just remove the mask_array to save space
    if output.clipped_percent == 0
        output.mask_array = [];
    end


    % subtract electrodes if there are 2
    % from here onwards, only channel_1 is used
    if ~isempty(channel_2)
        channel_1 = channel_1 - channel_2;
    end
    

    % get statistics on unique values in epochs
    % before any filtering is done (which will increase the count)
    channel_1_reshaped = reshape(channel_1, [], epochs);
    output.epoch_mad = mad(channel_1_reshaped, 1)';
    output.epoch_mad_z = robust_zscore(output.epoch_mad);
    
    output.zero_mad_epochs_percent = length(find(output.epoch_mad == 0))/length(output.epoch_mad)*100;
    output.low_mad_epochs_percent = length(find(output.epoch_mad_z < -3))/length(output.epoch_mad_z)*100;
    output.min_z_epoch_mad = min(output.epoch_mad_z);
    
    epoch_unique_values = zeros(epochs, 1, 'single');
    for epoch_i = 1:epochs
        epoch_unique_values(epoch_i) = length(unique(channel_1_reshaped(:, epoch_i)));
    end
    output.epoch_unique_values = epoch_unique_values;
    output.epochs_with_few_unique_values_percent = length(find(epoch_unique_values <= 8)) / epochs * 100;


    %
    % remove noise
    %
    
    % remove the baseline wander (< 0.5 Hz)
    hp_filter = get_hp_filter(fs);
    channel_1 = filtfilt(hp_filter, channel_1);

    % all recordings are from US, remove 60Hz (regardless)
    if fs > 120
        channel_1 = remove_noise(channel_1, fs, ecg_filter_bw, 60);
        output.noise_freqs = [60];
    elseif fs >= 70
        % if nyquist is below 60Hz, then 60 will be aliased below the
        % nyquist
        alias60 = fs - 60;
        channel_1 = remove_noise(channel_1, fs, ecg_filter_bw, alias60);
        output.noise_freqs = [alias60];
    else
        % a handful have an fs of 50
        output.noise_freqs = [];
    end


    % remove additional noise
    % for ccshs, cfs, and chat rarely were real regular noise found
    % if one-off fixes are needed later, I will apply
    % need to have enough to grab enough snapshots
    if epochs > 2
        [channel_1, additional_freqs] = find_and_remove_noise(channel_1, fs, ecg_min_freq, ecg_noise_min_ratio, ecg_filter_bw, max_additional_freqs);
    else
        additional_freqs = [];
    end
    
    % update the noise freq list
    output.noise_freqs = [output.noise_freqs; additional_freqs];
    
    %
    % resample to desired freq
    %
    if fs ~= desired_fs
        % get up/down factors as small as possible
        [p, q] = rat(desired_fs/fs);

        % resample and store
        output.ecg = resample(channel_1, p, q);
        output.fs = desired_fs;
    else
        % just store
        output.ecg = channel_1;
        output.fs = desired_fs;
    end
    
    %
    % normalize
    %
    output.ecg = robust_zscore(output.ecg);

    %
    % save file
    %
    save_struct(outputfilename, output, true, true);
    % ppm.increment();
end
toc

% delete(ppm)
disp('done')


%%




%% get stats on percents

% for i = 1:count
%     outputfilename = [files(i).folder, '/', files(i).name];
%     outputfilename = strrep(outputfilename, 'raw_ecgs', 'resampled_ecgs');
%     data = load(outputfilename, 'clipped_percent');
%     files(i).clipped_percent = data.clipped_percent;
% end



%% old channels used code
% % combinations (in order)
%     % ECG1-ECG2 - covered
%     % ECGL-ECGR - covered
%     % ECG1_2 - covered
%     % ECGLECGR - covered
%     % ECG1-ECG3  (missing lead2) - covered
%     % ECG2 (missing lead1 and 3) - covered
%     % ECG (only option) - covered
%     % EKG (only option) - covered
% 
%     if strcmp(ch_list{1}, 'ECG') & (length(ch_list) == 1)
%         % only an ECG channel, which looked acceptable
%         channel_1 = data.ECG.data;
%         channels_used = 'ECG';
% 
%     elseif strcmp(ch_list{1}, 'ECG2') & (length(ch_list) == 1)
%         % only ECG2 channel (weird one-off)
%         channel_1 = data.ECG2.data;
%         channels_used = 'ECG2';
% 
%     elseif strcmp(ch_list{1}, 'EKG')
%         % only an EKG channel, which looked acceptable
%         % could also have EKG_Off (so don't check ch_list length)
%         channel_1 = data.EKG.data;
%         channels_used = 'EKG';
% 
%     elseif strcmp(ch_list{1}, 'ECG1_2')
%         % these are always the only channel
%         channel_1 = data.ECG1_2.data;
%         channels_used = 'ECG1_2';
% 
%     elseif strcmp(ch_list{1}, 'ECGLECGR')
%         % these are always the only channel
%         channel_1 = data.ECGLECGR.data;
%         channels_used = 'ECGLECGR';
% 
%     elseif strcmp(ch_list{1}, 'ECG1') & strcmp(ch_list{2}, 'ECG2')
%         % preferred case
%         % sometimes there is an extra 'ECG' channel, but will assume labels
%         % are correct
%         channel_1 = data.ECG1.data;
%         channel_2 = data.ECG2.data;
%         channels_used = 'ECG1-ECG2';
% 
%     elseif strcmp(ch_list{1}, 'ECGL') & strcmp(ch_list{2}, 'ECGR')
%         channel_1 = data.ECGL.data;
%         channel_2 = data.ECGR.data;
%         channels_used = 'ECGL-ECGR';
% 
%     elseif strcmp(ch_list{2}, 'ECG3')
%         % no ECG2 channel, so use ECG1 and ECG3
%         channel_1 = data.ECG1.data;
%         channel_2 = data.ECG3.data;
%         channels_used = 'ECG1-ECG3';
% 
%     else
%         channels_used = 'ERROR';
%     end


%% old normalization code
% for normalization, but I think this should wait until later
% limit = 1;
% zscore_div = 50;
% %
    % % normalize
    % %
    % output.ecg = robust_zscore(output.ecg)/zscore_div;
    % output.ecg(output.ecg > limit) = limit;
    % output.ecg(output.ecg < -limit) = -limit;
    % 
    % % calculate total percent of outliers
    % outliers = (abs(output.ecg) >= limit);
    % output.outlier_percent = sum(outliers)/length(outliers)*100;
    % 
    % % get the epoch-specific outlier percent
    % outliers = reshape(outliers, (output.fs*30), [])';
    % output.epoch_outlier_percent = single(sum(outliers,2)/(output.fs*30)*100);


