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


function [y, first_index, last_index] = select_centered_window(y, window_length)
%cut a centered window

assert(isvector(y));
assert(window_length <= length(y), 'window_length cannot be longer than input');
assert(mod(window_length, 1) == 0, 'window_length must be an integer');

%get the centered window
center_index = (length(y) + mod(length(y), 2))/2;
first_index = center_index - (window_length + mod(window_length, 2))/2 + 1;
last_index = first_index + window_length - 1;

%check the indexes
assert(first_index >= 1);
assert(last_index <= length(y));
assert(mod(first_index, 1) == 0);
assert(mod(last_index, 1) == 0);

%cut center of y to max length
y = y(first_index:last_index);

end
