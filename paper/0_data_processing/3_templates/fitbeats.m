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


function [fitobjectout, org_locations_fit] = fitbeats(org_locations)

    org_locations_fit = org_locations;
    org_locations_fit(2:end, 2) = diff(org_locations_fit);
    
    org_locations_fit(org_locations_fit(:,2)<80, :) = [];
    org_locations_fit(org_locations_fit(:,2)>512, :) = [];
    
    % fit a low order poly first
    fitobjectout = fit(org_locations_fit(:,1), org_locations_fit(:,2),'poly2','Normalize','on','Robust','Bisquare');
    
    % predict those values
    org_locations_fit(:, 3) = feval(fitobjectout, org_locations_fit(:,1));
    org_locations_fit(:, 4) = org_locations_fit(:, 2)./org_locations_fit(:, 3);
    
    %
    % remove the ones that are greater than 75% over the prediction
    org_locations_fit(org_locations_fit(:, 4) > 1.75, :) = [];
    % org_locations_fit(org_locations_fit(:, 4) < 0.6, :) = [];
    
    
    % now fit using a higher order and plot
    fitobjectout = fit(org_locations_fit(:,1), org_locations_fit(:,2),'poly4','Normalize','on','Robust','Bisquare');
end