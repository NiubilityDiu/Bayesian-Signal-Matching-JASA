function [eeg_down] = extractRawEEGRCP(sub_name_short, raw_file_name, data_type, band_low, band_upp, dec_factor, local_bool)

% This function is used to access and pre-process K-protocol data, both TRN and FRT files.

if local_bool
    folder_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode/EEG_MATLAB_data/';
else
    folder_dir = '/home/mtianwen/EEG_MATLAB_data';
end

file_name = strcat(sub_name_short, '_', raw_file_name, '.mat');
file_dir = [folder_dir filesep data_type filesep sub_name_short ...
    filesep file_name];
mat_obj = load(file_dir);

if isfield(mat_obj, 'Data')
    mat_obj = mat_obj.Data;
else
    mat_obj = mat_obj.data;
end

% Raw Data
signal = mat_obj.RawData.signal;
states = mat_obj.RawData.states;
parameters = mat_obj.RawData.parameters;
channel_name = parameters.ChannelNames.Value;

% State variables
stim_code = states.StimulusCode;
switch data_type
    case 'TRN_files'
        stim_type = states.StimulusType;
    case 'FRT_files'
        stim_type = mat_obj.DBIData.DBI_EXP_Info.Stimulus_Type;
end
stim_begin = states.StimulusBegin;
stim_pis = states.PhaseInSequence;
intended_text = string(mat_obj.DBIData.DBI_EXP_Info.Intended_Text(:,2)); 
letter_table = mat_obj.RawData.parameters.TargetDefinitions.Value(:,1);

% remove confusion signal and parameters (only for testing files)
if ~strcmp(data_type, 'TRN')
    % idenfity the first zero value of PIS
    pis_first_zero_idx = find(stim_pis == 0, 1);
    
    pis_last_three_idx = find(stim_pis == 3, 1, 'last');
    pis_last_zero_idx = find(stim_pis == 0, 1, 'last');
    pis_last_one_idx = find(stim_pis == 1, 1, 'last');
    pis_last_idx = max([pis_last_three_idx pis_last_zero_idx pis_last_one_idx]);
    % Use max to extract the delayed response as much as possible.
    
    signal = signal(pis_first_zero_idx:pis_last_idx,:);
    stim_type = stim_type(pis_first_zero_idx:pis_last_idx,:);
    stim_begin = stim_begin(pis_first_zero_idx:pis_last_idx,:);
    stim_pis = stim_pis(pis_first_zero_idx:pis_last_idx,:);
    stim_code = stim_code(pis_first_zero_idx:pis_last_idx,:);
end

% Bandpass Filter
% band_low = 0.5;
fs = parameters.SamplingRate.NumericValue;
sprintf('Sampling rate is %d Hz.', fs);
signal_bp = bandpass(signal, [band_low, band_upp], fs);
% Spatial Filter (common average reference)
% signal_bp_mean = mean(signal_bp, 2);
% signal_bp = signal_bp - signal_bp_mean;

% Down-sampling Scheme
signal_bp_down = downsample(signal_bp, dec_factor);
stim_code_down = downsample(stim_code, dec_factor);
stim_type_down = downsample(stim_type, dec_factor);
stim_begin_down = downsample(stim_begin, dec_factor);
stim_pis_down = downsample(stim_pis, dec_factor);

eeg_down = struct();  % do not add [] inside parentheses.
eeg_down.Signal = signal_bp_down;
eeg_down.Type = stim_type_down;
eeg_down.Begin = stim_begin_down;
eeg_down.PIS = stim_pis_down;
eeg_down.Fs = fs;
eeg_down.ChannelName = channel_name;
eeg_down.Code = stim_code_down;
eeg_down.Text = intended_text;
eeg_down.LetterTable = letter_table;

end