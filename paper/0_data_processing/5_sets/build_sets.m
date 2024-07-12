clc
clear

%% load the data
df = readtable('dataframe_input.xlsx');

% remove the excluded records
df = df(~df.exclude, :);

%%

% get the list of dataset names
dataset_list = unique(df.dataset);

% get the counts for each subject_id
df_subject_ids = table();
df_subject_ids.subject_id = unique(df.subject_id);
df_subject_ids.subject_id_count = cellfun(@(x) sum(ismember(df.subject_id, x)), unique(df.subject_id));

% add the counts to the table
df = join(df, df_subject_ids);

clear df_subject_ids

%% load the desired counts
desired_counts = readtable('decade_counts.xlsx');
desired_val_counts = zeros(10,2);
desired_train_counts = zeros(10,2);
desired_val_counts(:,1) = desired_counts.val_female;
desired_val_counts(:,2) = desired_counts.val_male;
desired_train_counts(:,1) = desired_counts.train_female;
desired_train_counts(:,2) = desired_counts.train_male;
clear desired_counts

val_train_ratio = sum(desired_val_counts,'all')/(sum(desired_val_counts,'all') + sum(desired_train_counts,'all'));


%%

clc
fprintf('find sets\n');
% clear out the set
df.new_2023_set = repmat({''}, height(df), 1);

% set the seed
rng(12);

% since decade 3 and 4 are both used completely, and had a lot of dupe
% recordings, they need to be placed first in train and val
% decade 10 is also used, but it contains no dupe subjects
% proportionally between the two sets

fprintf('place decade 2-5 subjects with multiple recordings first\n');
for male = 0:1
    for decade = [2,3,4,5]
        subject_ids = unique(df((df.subject_id_count > 1) & (df.decade==decade) & (df.male==male), :).subject_id);
        
        % shuffle list
        subject_ids = subject_ids(randperm(length(subject_ids)));

        val_subject_count = round(length(subject_ids)*val_train_ratio);
        val_recording_count = height(df(ismember(df.subject_id, subject_ids(1:val_subject_count)), :));

        df(ismember(df.subject_id, subject_ids(1:val_subject_count)), :).new_2023_set = repmat({'validation'}, val_recording_count, 1);
        
        train_subject_count = length(subject_ids) - val_subject_count;
        train_recording_count = height(df(ismember(df.subject_id, subject_ids((val_subject_count+1):end)), :));

        df(ismember(df.subject_id, subject_ids((val_subject_count+1):end)), :).new_2023_set = repmat({'train'}, train_recording_count, 1);
    end
end


% now, fill all of the sets
fprintf('place all other recordings now\n');

% going in "reverse" order, because it is hard to fill the smaller sets
for setnamecell = {'test', 'validation', 'train'}
    setname = setnamecell{1};
    fprintf('start %s set\n', setname);
    for male = 0:1
        for decade = 1:10
            counts = get_decade_counts_for_set(df, setname);
            if ~strcmp(setname, 'train')
                remaining = desired_val_counts(decade, male+1) - counts(decade, male+1);
            else
                remaining = desired_train_counts(decade, male+1) - counts(decade, male+1);
            end
    
            while (remaining > 0)
                
                if ~strcmp(setname, 'train')
                    % for test or validation set, just prevent using
                    % subjects from the test set
                    % get the list of subjects in the test set
                    list_of_subject_ids = get_subject_ids_in_set(df, 'test');
                else
                    % for train set, prevent using subjects in test or
                    % validation set
                    % get the list of subjects in the test and validation set
                    list_of_subject_ids_test = get_subject_ids_in_set(df, 'test');
                    list_of_subject_ids_validation = get_subject_ids_in_set(df, 'validation');
                    list_of_subject_ids = unique([list_of_subject_ids_test; list_of_subject_ids_validation]);
                end
                
                % get list of orders that are available and not in the current set
                if strcmp(setname, 'test')
                    % for test set, only use unique subjects
                    % orders = df((df.subject_id_count==1) & (df.male==male) & (df.decade==decade) & strcmp(df.new_2023_set, '') & (~ismember(df.subject_id, list_of_subject_ids)), :).order;
                    % orders = df((df.male==male) & (df.decade==decade) & strcmp(df.new_2023_set, '') & (~ismember(df.subject_id, list_of_subject_ids)), :).order;
                    
                    % exclude old train/validation records from test
                    orders = df((~(strcmp(df.old_2018_set, 'train') | strcmp(df.old_2018_set, 'validation'))) & (df.male==male) & (df.decade==decade) & strcmp(df.new_2023_set, '') & (~ismember(df.subject_id, list_of_subject_ids)), :).order;
                else
                    orders = df((df.male==male) & (df.decade==decade) & strcmp(df.new_2023_set, '') & (~ismember(df.subject_id, list_of_subject_ids)), :).order;
                end
                
                % there should be at least 1 option
                if isempty(orders)

                    fprintf('exited early %i %i\n', male, decade);
                    break
                end
        
                % shuffle the list
                orders = orders(randperm(length(orders)));
                
                % take the first, and add it to the set
                df(df.order == orders(1), :).new_2023_set = {setname};
                
                % get the updated counts
                counts = get_decade_counts_for_set(df, setname);
                if ~strcmp(setname, 'train')
                    remaining = desired_val_counts(decade, male+1) - counts(decade, male+1);
                else
                    remaining = desired_train_counts(decade, male+1) - counts(decade, male+1);
                end
    
                %fprintf('%i %i %i %i\n', male, decade, remaining, height(df(strcmp(df.new_2023_set, setname), :)))
            end
        end
    end
    fprintf('  done with %s set\n', setname);
end

% get the counts left
remaining_train_counts = desired_train_counts - get_decade_counts_for_set(df, 'train');
assert(sum(remaining_train_counts, 'all') == 0)
remaining_validation_counts = desired_val_counts - get_decade_counts_for_set(df, 'validation');
assert(sum(remaining_validation_counts, 'all') == 0)
remaining_test_counts = desired_val_counts - get_decade_counts_for_set(df, 'test');
assert(sum(remaining_test_counts, 'all') == 0)

disp('done')


%%

clc
fprintf('swap recordings to balance datasets\n');

setnames = {'train','validation','test'};

% get the initial error
dataset_error = get_dataset_error(df, dataset_list);
fprintf('error: %.3f\n', dataset_error);

% misc
previous_error = dataset_error;
miss_count = 0;
max_miss_count = 2000;

while 1
    df_temp = random_swap_recordings(df, dataset_list, setnames);
    dataset_error = get_dataset_error(df_temp, dataset_list);

    if dataset_error < previous_error
        previous_error = dataset_error;
        fprintf('error: %.3f   misses: %i\n', dataset_error, miss_count)
        df = df_temp;
        miss_count = 0;
    else
        miss_count = miss_count + 1;
        if miss_count > max_miss_count
            break
        end
    end
end

% update results
[dataset_error, dataset_counts, dataset_ratios] = get_dataset_error(df, dataset_list);

% check that each subject appears in only one set
assert(length(unique(df(ismember(df.subject_id, unique(df((strcmp(df.new_2023_set, 'train')) , :).subject_id)) & (~strcmp(df.new_2023_set, '')), :).new_2023_set)) == 1)
assert(length(unique(df(ismember(df.subject_id, unique(df((strcmp(df.new_2023_set, 'validation')) , :).subject_id)) & (~strcmp(df.new_2023_set, '')), :).new_2023_set)) == 1)
assert(length(unique(df(ismember(df.subject_id, unique(df((strcmp(df.new_2023_set, 'test')) , :).subject_id)) & (~strcmp(df.new_2023_set, '')), :).new_2023_set)) == 1)

disp('done')


%%

age_train = mean(df(strcmp(df.new_2023_set, "train"), :).age);
age_val = mean(df(strcmp(df.new_2023_set, "validation"), :).age);
age_test = mean(df(strcmp(df.new_2023_set, "test"), :).age);
fprintf('%.2f  %.2f  %.2f\n', age_train, age_val, age_test);

%%

for i = 1:length(dataset_list)
    df(strcmp(df.dataset, dataset_list{i}), :).dataset_name_id = i*ones(height(df(strcmp(df.dataset, dataset_list{i}), :)), 1);
end



%%

% start with male column
for i = 12:length(df.Properties.VariableNames)
    var = df.Properties.VariableNames{i};
    try
        ax = histogram(df.(var), "Normalization","probability");
        hold on

        % histogram(df(~strcmp(df.new_2023_set, '') , :).(var), "Normalization","probability", "DisplayStyle","stairs");
        linewidth = 2;
        histogram(df(strcmp(df.new_2023_set, 'train') , :).(var), "Normalization","probability", "DisplayStyle","stairs", "BinWidth",ax.BinWidth, "BinLimits",ax.BinLimits, "LineWidth",linewidth);
        histogram(df(strcmp(df.new_2023_set, 'validation') , :).(var), "Normalization","probability", "DisplayStyle","stairs", "BinWidth",ax.BinWidth, "BinLimits",ax.BinLimits, "LineWidth",linewidth);
        histogram(df(strcmp(df.new_2023_set, 'test') , :).(var), "Normalization","probability", "DisplayStyle","stairs", "BinWidth",ax.BinWidth, "BinLimits",ax.BinLimits, "LineWidth",linewidth);
        hold off
        % ylim([0, 1])
        title(strrep(var,"_"," "));
        saveas(gcf, ['distribution_', var, '.png']);
        % pause
        
    catch
    end

end





%% helper functions

function [decade_counts] = get_decade_counts(df)
    % two columns, female, male and 10 rows
    decade_counts = zeros(10, 2);
    decade_counts(:, 2) = histcounts(df(df.male==1, :).decade, 1:11)';
    decade_counts(:, 1) = histcounts(df.decade, 1:11)' - decade_counts(:, 2);
end

function [decade_counts] = get_decade_counts_for_set(df, setname)
    % two columns, female, male and 10 rows
    decade_counts = zeros(10, 2);
    decade_counts(:, 2) = histcounts(df((df.male==1) & strcmp(df.new_2023_set, setname), :).decade, 1:11)';
    decade_counts(:, 1) = histcounts(df(strcmp(df.new_2023_set, setname), :).decade, 1:11)' - decade_counts(:, 2);
end

function [list_of_subject_ids] = get_subject_ids_in_set(df, setname)
    list_of_subject_ids = unique(df(strcmp(df.new_2023_set, setname),  :).subject_id);
end

function [orders] = get_available_orders(df)
    orders = df(strcmp(df.new_2023_set, ''), :).order
end


function [counts] = get_set_dataset_counts(df, setname, dataset_list)
    counts = zeros(length(dataset_list), 1);
    for i = 1:length(dataset_list)
        counts(i) = height(df(strcmp(df.new_2023_set, setname) & strcmp(df.dataset, dataset_list{i}), :));
    end
end


function [dataset_error, dataset_counts, dataset_ratios] = get_dataset_error(df, dataset_list)
    % get the ratio for each dataset in each set (try to normalize this)
    dataset_counts = zeros(length(dataset_list), 3);
    dataset_counts(:, 1) = get_set_dataset_counts(df, 'train', dataset_list);
    dataset_counts(:, 2) = get_set_dataset_counts(df, 'validation', dataset_list);
    dataset_counts(:, 3) = get_set_dataset_counts(df, 'test', dataset_list);
    
    % normalize by the column sum
    dataset_ratios = dataset_counts./sum(dataset_counts);
    dataset_per_error = (dataset_ratios - mean(dataset_ratios,2))./mean(dataset_ratios,2);
    
    % get the error as the sum of the std/mean of each dataset
    % this is the goal to minimize
    dataset_error = sum(std(dataset_ratios, [], 2)./mean(dataset_ratios, 2));
end

function [df, error_code] = random_swap_recordings(df, dataset_list, setnames)
    
    setname_from = setnames{randi(length(setnames))};    
    setname_to = setnames{randi(length(setnames))};
    while strcmp(setname_to, setname_from)
        setname_to = setnames{randi(length(setnames))};
    end
    
    dataset_from = dataset_list{randi(length(dataset_list))};
    dataset_to = dataset_list{randi(length(dataset_list))};
    while strcmp(dataset_to, dataset_from)
        dataset_to = setnames{randi(length(setnames))};
    end

    % fprintf('%s %s  %s %s\n', setname_from, dataset_from, setname_to, dataset_to);
    
    % only allow swapping subjects that have a single recording (even if
    % only one recording is actually selected)
    
    if strcmp(setname_to, 'test')
        options_from = unique(df((~(strcmp(df.old_2018_set, 'train') | strcmp(df.old_2018_set, 'validation'))) & strcmp(df.dataset, dataset_from) & strcmp(df.new_2023_set, setname_from) & (df.subject_id_count == 1) , {'male','decade'}));    
    else
        options_from = unique(df(strcmp(df.dataset, dataset_from) & strcmp(df.new_2023_set, setname_from) & (df.subject_id_count == 1) , {'male','decade'}));    
    end
    
    options_to = unique(df(strcmp(df.dataset, dataset_to) & strcmp(df.new_2023_set, setname_to) & (df.subject_id_count == 1), {'male','decade'}));
    options_join = innerjoin(options_from, options_to);
    
    % shuffle join
    options_join = options_join(randperm(height(options_join)), :);
    
    if ~isempty(options_join)

        % get orders from and shuffle
        if strcmp(setname_to, 'test')
            % exclude old train/validation records from test
            orders_from = df((~(strcmp(df.old_2018_set, 'train') | strcmp(df.old_2018_set, 'validation'))) & strcmp(df.dataset, dataset_from) & strcmp(df.new_2023_set, setname_from) & (df.subject_id_count == 1) & (df.male == options_join(1,:).male) & (df.decade == options_join(1,:).decade),:).order;
        else
            orders_from = df(strcmp(df.dataset, dataset_from) & strcmp(df.new_2023_set, setname_from) & (df.subject_id_count == 1) & (df.male == options_join(1,:).male) & (df.decade == options_join(1,:).decade),:).order;
        end
        orders_from = orders_from(randperm(length(orders_from)));
        
        % get orders to and shuffle
        if strcmp(setname_from, 'test')
            % exclude old train/validation records from test
            orders_to = df((~(strcmp(df.old_2018_set, 'train') | strcmp(df.old_2018_set, 'validation'))) & strcmp(df.dataset, dataset_to) & strcmp(df.new_2023_set, setname_to) & (df.subject_id_count == 1) & (df.male == options_join(1,:).male) & (df.decade == options_join(1,:).decade),:).order;
        else
            orders_to = df(strcmp(df.dataset, dataset_to) & strcmp(df.new_2023_set, setname_to) & (df.subject_id_count == 1) & (df.male == options_join(1,:).male) & (df.decade == options_join(1,:).decade),:).order;
        end
        orders_to = orders_to(randperm(length(orders_to)));
        
        % fprintf('from: %i  to: %i\n', length(orders_from), length(orders_to));
        
        if ~isempty(orders_from) & ~isempty(orders_to)
            df(df.order == orders_from(1), :).new_2023_set = {setname_to};
            df(df.order == orders_to(1), :).new_2023_set = {setname_from};
            
            % fprintf('done\n');
            error_code = 0;
        else
            % fprintf('no matching orders\n');
            error_code = 1;
        end
    else
        % fprintf('no matching options\n');
        error_code = 2;
    end
end
