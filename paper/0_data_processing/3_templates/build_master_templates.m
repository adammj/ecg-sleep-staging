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



all_templates = zeros(9562, 256);

for i = 1:9562
    all_templates(i,:) = files(i).r_wave_template;
end


%%
remaining_templates = all_templates;

%%
% first choose the first template and get the correlations with it
corr_thres = 0.5;

count = size(remaining_templates, 1);
ppm = ParforProgressbar(count, 'showWorkerProgress', true);

counts = zeros(count, 1);
for j = 9 %:count
    %fprintf('%i/%i\n', j, size(remaining_templates, 1));
    % j=1;

    template = remaining_templates(j, :);
    previous_high_corr_i = [];
    
    for i = 1:50
        all_corrs = corr(remaining_templates', template');
        high_corr_i = find(abs(all_corrs) >= corr_thres);
        
        % get polarity based on corr sign
        polarity = sign(all_corrs(high_corr_i));
        
        % get the possible templates
        possible_templates = remaining_templates(high_corr_i, :) .*  polarity;
        
        % create the template
        template = median(robust_zscore(possible_templates')', 1);
        %fprintf('%i: %i\n', i, length(high_corr_i));
        
        % if the same templates were selected again, then stop
        if ~isempty(previous_high_corr_i) & isequal(high_corr_i, previous_high_corr_i)
            break
        end
    
        previous_high_corr_i = high_corr_i;
    end

    counts(j) = length(high_corr_i);
    ppm.increment();
end
disp('done')
delete(ppm)

%%

% now, remove the rows used to match to the template
remaining_templates(high_corr_i, :) = [];


%%
for i = 1:size(remaining_templates,1 )
    disp(i)
    plot(remaining_templates(i,:))
    pause
end


%%
probable_subject = zeros(8,2);
for i = 1:8
    best_corr = 0;
    best_corr_id = 0;
    for j = 1:2884
        corr_value = corr(files(j).r_wave_template', remaining_templates(i,:)');
        if corr_value > best_corr
            best_corr = corr_value;
            best_corr_id = j;
        end
    end

    probable_subject(i, :) = [best_corr_id, best_corr];
end



%%
for i = 1:size(remaining_templates, 1)
    plot(remaining_templates(i,:))
    fprintf('%i\n',i);
    pause
end



%%

template_matches = zeros(9562, size(archetype_templates, 1));
for i = 1:size(archetype_templates, 1);
    template_matches(:, i) = abs(corr(all_templates', archetype_templates(i,:)'));
end





