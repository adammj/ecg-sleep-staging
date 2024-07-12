



%%
clc
count_files = height(files_df);
ppm = ProgressBar(count_files);

input_folder = '/sleep_data/resampled_ecgs/';
output_folder = '/sleep_data/datasets/dataset_multirater_files/';

parfor i = 1:count_files

    filename_input = [input_folder, files_df(i, :).dataset{1}, '/', files_df(i, :).source_file{1}, '.mat'];
    filename_output = [output_folder, files_df(i, :).source_file{1}, '.mat'];
    
    if files_df(i, :).done || isfile(filename_output)
        %disp('exists');
        files_df(i, :).done = 1;
        count(ppm);
        continue
    end
    

    % try
        % load all of the data, it will be modified in-place
        data = load(filename_input);
        
    
        %
        % add stages to data
        %
        %stage_i = find(strcmp({all_stages.file}, files_df(i, :).source_file{1}));
        %data.stages = all_stages(stage_i).stages;
        data.stages = zeros(data.epochs, 1);
    
    
        %
        % remove unnecessary (large) fields for dataset
        %
        if isfield(data, 'template')
            data = rmfield(data, 'template');
            data = rmfield(data, 'template_locations');
        end
        data = rmfield(data, 'mask_array');
        
        % store the filename for use by the loader
        data.filename = files_df(i, :).source_file{1};
    
        %
        % calculate weight
        %
        data.weights = ones(data.epochs, 1, 'single');
        data.weights_reason = zeros(data.epochs, 1, 'uint8');
        
        % weight scaled by epoch_nonclipped_percent
        % any epoch that was downscaled gets a reason=1
        % those with 100% clipped will recieve a weight of 0
        data.epoch_nonclipped_ratio = (100 - data.epoch_clipped_percent)/100;
        data.weights = data.weights .* data.epoch_nonclipped_ratio;
        data.weights_reason(data.epoch_nonclipped_ratio < 1) = 1;
        
        % now, use the robust zscore mad
        % any epoch that contains a zscored mad < -3 gets 0.5 * the current
        % weight, and a reason=2
        % data.epoch_mad_z = robust_zscore(data.epoch_mad);
        % data.weights(data.epoch_mad_z < -3) = 0.5 * data.weights(data.epoch_mad_z < -3);
        % data.weights_reason(data.epoch_mad_z < -3) = 2;
    
        % now, use the the original mad values, and any epochs with a value = 0
        % get a weight of 0 and a reason=3
        % data.weights(data.epoch_mad <= 0) = 0;
        % data.weights_reason(data.epoch_mad <= 0) = 3;
        
        % use unique values for each epoch (more precise for eliminiating
        % epochs (weight=0))
        data.weights(data.epoch_unique_values <= 8) = 0;
        data.weights_reason(data.epoch_unique_values <= 8) = 2;
    
        % finally, for the epochs that were unscored, give them a weight of 0
        % and reason=4
        % stages is currently untransformed (so REM = 5, and anything higher is
        % unscored)
        data.weights(data.stages > 5) = 0;
        data.weights_reason(data.stages > 5) = 4;
        
        % check weights
        assert(isempty(find(data.weights < 0)))
        assert(isempty(find(data.weights > 1)))
    
    
        %
        % create the demographics
        %
        data.demographics = single([files_df(i, :).male; files_df(i, :).age/100]);
        
        % check demographics
        assert(data.demographics(1) >= 0)
        assert(data.demographics(1) <= 1)
        assert(data.demographics(2) >= 0)
        assert(data.demographics(2) <= 1)
    
    
        %
        % convert the stages
        %
        data.stages(data.stages == 4) = 3;  % move S4 to 3 (SWS=S3+S4)
        data.stages(data.stages == 5) = 4;  % move REM to 4
        data.stages(data.stages > 5) = 0;   % make all unscored epochs W (but they also have a weight of 0)
        
        % check stages
        assert(isempty(find(data.stages < 0)))
        assert(isempty(find(data.stages > 4)))
    
    
        %
        % normalize and clip ecg to +/-1
        %
        %data.ecg = data.ecg ./ files_df(i, :).normalization_factor;
        data.ecg = data.ecg ./ data.normalization_factor;
        data.ecg(data.ecg > 1) = 1;
        data.ecg(data.ecg < -1) = -1;
    
        % check ecg
        assert(isempty(find(data.ecg < -1)))
        assert(isempty(find(data.ecg > 1)))
    
        
        %
        % midnight offset
        %
        % this is the start_time relative to midnight
        % so, if it starts before midnight (but after noon), then it will be negative
        % if it starts before noon, then it will be positive
        
        % if after noon
        if duration(data.start_time) > duration('12:00:00')
            data.midnight_offset = seconds((duration(data.start_time) - duration('24:00:00')))/(24*3600);
        else
            data.midnight_offset = seconds((duration(data.start_time) - duration('00:00:00')))/(24*3600)
        end
        data.midnight_offset = single(data.midnight_offset);
    
        % check offset
        assert(data.midnight_offset <= 0.5)
        assert(data.midnight_offset >= -0.5)
    
        
        %
        % reshape the matrices that will be used by numpy into 
        % the same format used previously
        %
        data.weights_reason = data.weights_reason';
        data.weights = data.weights';
        data.stages = data.stages';
        % reshape and change the variable name for the ecg
        data.ecgs = reshape(data.ecg, (30*256), []);
        % delete the old variable
        data = rmfield(data, 'ecg');
    
        
        % final check on epoch dimension
        assert(size(data.ecgs, 2) == data.epochs)
        assert(size(data.weights, 2) == data.epochs)
        assert(size(data.stages, 2) == data.epochs)
    
    
        %
        % remove remaining unnecessary large fields for dataset
        %
        data = rmfield(data, 'epoch_clipped_percent');
        data = rmfield(data, 'epoch_nonclipped_ratio');
        data = rmfield(data, 'epoch_mad');
        data = rmfield(data, 'epoch_mad_z');
        data = rmfield(data, 'epoch_template_counts');
        data = rmfield(data, 'epoch_unique_values');
        
        save_struct(filename_output, data, true, true);
        files_df(i, :).done = 1;
    % catch
    % end
    count(ppm);
end

delete(ppm)
disp('done')

