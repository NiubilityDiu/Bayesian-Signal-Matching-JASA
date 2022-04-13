function [weight_obj] = trainSWLDAmatlabR(local_use, sub_new, data_type, sample_size_new, max_var_len)
%trainSWLDAmatlabR the script to run the matlab code in R for convenience.
local_use = str2num(local_use);
switch local_use
    case 1
        parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'; 
    otherwise 
        parent_dir = '/home/mtianwen'; 
end
rng(612);
eeg_data_dir = [parent_dir filesep 'EEG_MATLAB_data' filesep data_type ...
    filesep sub_new];
eeg_data_obj = load([eeg_data_dir filesep sprintf('%s_001_BCI_TRN_Truncated_Data.mat', sub_new)]);

signal = eeg_data_obj.Signal;
type = eeg_data_obj.Type;

sample_size_new = str2num(sample_size_new);
max_var_len = str2num(max_var_len);
signal_sub = signal(1:sample_size_new, :);
type_sub = type(1:sample_size_new, :);

[b, ~, pval, inmodel, ~] = trainSWLDAmatlab(signal_sub, type_sub, max_var_len);

% save swlda wts obj
weight_new_dir = [eeg_data_dir filesep 'swLDA' filesep 'Weight_001_BCI_TRN_new.mat'];
% make the output consistent
ID = sub_new;
sample_size = sample_size_new;

save(weight_new_dir, 'ID', 'b', 'pval', 'inmodel', 'sample_size');

% weight_obj = struct();
% weight_obj.ID = sub_new;
% weight_obj.b = b;
% weight_obj.pval = pval;
% weight_obj.inmodel = inmodel;
% weight_obj.sample_size = sample_size_new;
end