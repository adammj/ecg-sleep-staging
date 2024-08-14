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


function [autocorr] = smooth_autocorr(autocorr, constants)
%smooth out the autocorrelation (above lowest lag)
%brings in some influence from the 2 overlapping epochs on each side
%this is done because the epoch windows are overlapping, so information
%about each adjacted epoch is relevant to the current epoch

%create copy of epochs next to each end
temp_epoch_1 = autocorr(2, :);
temp_epoch_2 = autocorr((end-1), :);

%scale middle
autocorr(2:(end-1), :) = (autocorr(2:(end-1), :) + constants.smooth_nearby_scale*autocorr(1:(end-2), :) + constants.smooth_nearby_scale*autocorr(3:end, :)) / ( 1 + 2*constants.smooth_nearby_scale);

%scale ends
autocorr(1, :) = (autocorr(1, :) + constants.smooth_nearby_scale*temp_epoch_1) / (1+constants.smooth_nearby_scale);
autocorr(end, :) = (autocorr(end, :) + constants.smooth_nearby_scale*temp_epoch_2) / (1+constants.smooth_nearby_scale);

end
