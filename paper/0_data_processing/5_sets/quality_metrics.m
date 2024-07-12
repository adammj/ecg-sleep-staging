

clc

for i = 1:length(files)
    filename = [files(i).folder, '/', files(i).name];
    filename = strrep(filename, 'raw_ecgs', 'resampled_ecgs');
    % data = load(filename, "ecg");
    % 
    % plot(data.ecg)
    % disp(i)
    % pause

    % add missing variables to files
    % template = files(i).r_wave_template;
    % template_locations = files(i).locations;
    % normalization_factor = files(i).norm_factor;
    % save(filename, "template", "template_locations", "normalization_factor", "-append");
    
    data = load(filename, "stages");
    % files(i).epoch_mad = data.epoch_mad;
    %files(i).clipped_percent = data.clipped_percent;
    %files(i).noise_count = length(data.noise_freqs);

    files(i).wake = length(find(data.stages == 0));
    files(i).s1 = length(find(data.stages == 1));
    files(i).s2 = length(find(data.stages == 2));
    files(i).s3 = length(find(data.stages == 3));
    files(i).s4 = length(find(data.stages == 4));
    files(i).rem = length(find(data.stages == 5));
    files(i).unknown = length(find(data.stages > 5));
end

disp('done')
%%


for i = 1:length(files)
    epoch_mad_z = robust_zscore(files(i).epoch_mad);
    files(i).min_epoch_mad_z = min(epoch_mad_z);

    % can't use a ratio, as the min might be too high
    mad_thresh = -3; %min(epoch_mad_z) + (0 - min(epoch_mad_z))*0.1;
    files(i).low_mad_epochs_percent = length(find(epoch_mad_z <= mad_thresh))/length(epoch_mad_z)*100;
    files(i).zero_epoch_mad = length(find(files(i).epoch_mad == 0))/length(epoch_mad_z)*100;
end



%%
previous_j = 1;

for i = 4941:7510
    if mod(i, 100) == 0
        fprintf('%i\n', i);
    end

    filename = [files(i).folder, '/', files(i).name];
    filename = strrep(filename, 'raw_ecgs', 'resampled_ecgs');
    
    basename = strrep(files(i).name, '.mat', '');
    basename = strrep(basename, '-nsrr', '');
    
    found_j = 0;
    for j = previous_j:15211
        if strcmp(basename, all_dataset_stages(j).source)
            found_j = j;
            break
        end
    end
    if found_j == 0
        % start from the beginning
        for j = 1:15211
            if strcmp(basename, all_dataset_stages(j).source)
                found_j = j;
                break
            end
        end
    end
    
    % for those with the tricky stages
    if found_j == 0
        stages = single([]);
    else
        previous_j = found_j;
        stages = all_dataset_stages(found_j).stages;
    end

    save(filename, "stages", "-append");
end
disp('done')