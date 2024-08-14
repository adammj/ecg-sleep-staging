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

files = dir('*.mat');

ppm = ParforProgressbar(length(files), 'showWorkerProgress', true);

parfor i = 1:length(files)
    data = load(files(i).name);
    
    % if the field exists, then modify
    if isfield(data, 'data')
        
        % add the channels
        for j = 1:length(data.data)
            data.(data.data(j).channel) = data.data(j);
        end
        
        % remove the main
        data = rmfield(data, 'data');
        
        % replace the file
        save_struct(files(i).name, data, true, true);
    end
    
    ppm.increment();
end

delete(ppm)