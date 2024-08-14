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


function [constants] = get_constants(fs)


%% basic structure
constants = struct();


%% basic constants
constants.min_rr = 500;   %ms, or 120bpm (well above normal high HR during sleep)
constants.max_rr = 2000;  %ms, or  30bpm (well below normal low HR)
constants.fs = fs;


%% ecg/input filtering
constants.ecg.clip_fill_gap_length = round(1.0*fs);
constants.ecg.clip_smooth_transition_length = round(0.5*fs);
constants.ecg.clip_length = next_odd(0.3*fs);
constants.ecg.clip_ratio = 0.5;    %ratio of values spent at a limit, to indicate a square
constants.ecg.noise_min_freq = 4;
constants.ecg.noise_min_ratio = 4;  %ratios above this tend to show up in the autocorr
constants.ecg.noise_filter_bw = 1;
constants.ecg.hp_filter = get_hp_filter(fs);
constants.ecg.snapshot_count = 10;
constants.ecg.snapshot_freq_div = 0.5;

%% local_variation
constants.local_variation.local_samples = max(5, next_odd(0.060*fs));    %local variation local_samples becomes an issue below 5 samples
constants.local_variation.group_samples = next_odd(1.000*fs);  %maybe a little smaller?
constants.local_variation.min_peak_height = 1.5;
constants.local_variation.min_peak_separation = round(0.030*fs);
%constants.local_variation.avg_rr_max_rsd = 0.1; %unsure what this is
constants.local_variation.lp_filter = get_lp_filter(fs);


%% autocorrelation
%lag calculations
lowest_lag = floor(constants.min_rr/1000*fs);
highest_lag = ceil(constants.max_rr/1000*fs);

%lag of i1 to i2 (normal autocorr), i2 to i3 (one-at-a-time autocorr)
%include lag=0 so that perfect correlation is known
constants.autocorr.bounds = [0; (lowest_lag - 1); highest_lag];
%i1-i2 valid RR range
constants.autocorr.peak_bounds = [lowest_lag; highest_lag];
%i1 bottom to top of noise
constants.autocorr.noise_bounds = [floor(0.1*fs); floor(0.5*fs); highest_lag];
constants.autocorr.min_peak_corr = 0.05;	% percent correlation
constants.autocorr.min_peak_separation = round(0.08*fs);
constants.autocorr.harmonic_ratio = 0.8;    % consider valid harmonic if at least this ratio of the initial heigh
constants.autocorr.harmonic_max_offset = round(0.03*fs);  %+/- the calculated harmonic
constants.autocorr.smooth_nearby_scale = 0.25;
constants.autocorr.harmonic_lag_plus_minus = 3;
constants.autocorr.wide_max_scale_width = 7;    %the wider, the more sections are not scaled to 1


%% epochs
constants.epochs.window_s = 2*constants.max_rr/1000;  % was 5sec
constants.epochs.overlap_s = (constants.epochs.window_s)/2;  % was 2.5sec
constants.epochs.autocorr_quiet_threshold = 0.002;
constants.epochs.autocorr_noise_threshold = 0.110;
constants.epochs.autocorr_tall_peak_ratio = 0.25;	% to count the number of peaks this ratio of the max
constants.epochs.autocorr_max_tall_peaks = 4;       % max tall peaks allowed
constants.epochs.autocorr_peak_count_ratio = 0.50;	% percent of largest peak to count others
constants.epochs.group_width = next_odd(5);    % must be odd
constants.epochs.close_c1 = 5;      % emperical testing
constants.epochs.close_c2 = -0.2;   % emperical testing

%set known bounds
% epochs_bounds = get_epoch_bounds(length(analysis_struct.ecg.signal), fs, constants.epochs.window_s, constants.epochs.overlap_s);
% analysis_struct.epochs.epoch_count = size(epochs_bounds, 1);
% analysis_struct.epochs.bounds = epochs_bounds;

%% autocorrelation (again)
constants.autocorr.wide_width = next_odd(120/constants.epochs.overlap_s-constants.epochs.overlap_s); % 2 min, using correct overlap


%% r_indexes
% constants.r_waves.compare_samples = get_compare_samples_array(fs);

end


function [hp_filter] = get_hp_filter(fs)
persistent hp_filters

hp_filter = [];
if ~isempty(hp_filters)
    for i = 1:size(hp_filters, 1)
        if hp_filters(i, 1).fs == fs
            hp_filter = hp_filters(i).filter;
            break;
        end
    end
end

if isempty(hp_filter)
    hp_filter = designfilt('highpassiir', 'StopbandFrequency', 0.5, 'PassbandFrequency', 1, 'StopbandAttenuation', 30, 'PassbandRipple', 1, 'SampleRate', fs, 'DesignMethod', 'butter', 'MatchExactly', 'passband');
    
    %append to end
    hp_filters(end+1, 1).fs = fs;
    hp_filters(end, 1).filter = hp_filter;
end

end

function [lp_filter] = get_lp_filter(fs)
persistent lp_filters

lp_filter = [];
if ~isempty(lp_filters)
    for i = 1:size(lp_filters, 1)
        if lp_filters(i, 1).fs == fs
            lp_filter = lp_filters(i).filter;
            break;
        end
    end
end

if isempty(lp_filter)
    lp_filter = designfilt('lowpassiir', 'PassbandFrequency', 8, 'StopbandFrequency', 10, 'StopbandAttenuation', 30, 'PassbandRipple', 1, 'SampleRate', fs, 'DesignMethod', 'butter', 'MatchExactly', 'passband');
    
    %append to end
    lp_filters(end+1, 1).fs = fs;
    lp_filters(end, 1).filter = lp_filter;
end

end
