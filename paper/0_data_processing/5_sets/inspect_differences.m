

clc
df_used_in_2018 = df(~strcmp(df.x2018_set,''), :);
% odiff = zeros(919,2);

% good_stages = [];
% bad_stages = [];
demo_diff=zeros(919,2);

for i = 100 %:919
    % old_id = df_used_in_2018(i, :).x2018_id;
    % old_offset = sets_matrix(sets_matrix(:,1)==old_id, 15);
    % assert(~isempty(old_offset))
    % odiff(i, 1) = old_offset;
    
    filename_old = ['/sleep_data/dataset_1_files/', num2str(df_used_in_2018(i, :).x2018_id), '.mat'];
    data_old = load(filename_old );

    filename_new = ['/sleep_data/dataset_2023_files/', df_used_in_2018(i, :).source_file{1}];
    data_new = load(filename_new );
    % odiff(i, 2) = data.midnight_offset;
    % assert(length(data_old.stages) == length(data_new.stages))
    
    % if sum(data_old.stages~=data_new.stages) == 0
    %     good_stages = [good_stages; i];
    % else
    %     bad_stages = [bad_stages; i];
    % end

    % demo_diff(i, :) = (data_old.demographics - data_new.demographics)';
    %demo_diff(i,1) = mean(data_old.weights);
    %demo_diff(i,2) = mean(data_new.weights);

    middle_epoch = round(size(data_old.ecgs,2)/2);
    epochs = [1,middle_epoch, size(data_old.ecgs,2)];
    disp(i)
    for j = epochs

        plot((1:6000)/6000, data_old.ecgs(:,j))
        hold on
        plot((1:7680)/7680, data_new.ecgs(:,j))
        hold off
        pause
    end


end


%%
all_stages = zeros(4000, 5);
for i = 1:4000
    filename_new = ['/sleep_data/dataset_2023_files/', df(i, :).source_file{1}];
    data = load(filename_new, 'stages');
    all_stages(i, :) = histcounts(data.stages, 0:5);

end

