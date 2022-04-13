clear
clc

%%
K_nums = [106, 107, 108, ...
	  111, 112, 113, 114, 115, 117, 118, 119, 120, ...
	  121, 122, 123, ...
	  143, 145, 146, 147, ...
	  151, 152, 154, 155, 156, 158, 159, 160, ...
	  166, 167, ...
	  171, 172, 177, 178, 179, ...
	  183, 184, 185, 190, 191, 212, 223];

% K_num = K_nums(str2num(getenv('SLURM_ARRAY_TASK_ID')));
K_num = 117;
% disp(K_num);

dec_factor = 8;
bp_upp = 6;
zeta_binary_threshold = 0.3;

serial_num = '001';
exp_name = 'BCI';
data_type = 'TRN';
file_subscript = 'down';

if bp_upp < 0
    eeg_file_subscript = 'raw';
else
    eeg_file_subscript = strcat('raw_bp_0.5_', num2str(bp_upp));
end

% folder_dir = '/home/mtianwen/EEG_MATLAB_data';
folder_dir = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data';
% folder_dir = 'K:/Dissertation/Dataset and Rcode/EEG_MATLAB_data';

cd(strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num)));

%addpath /home/mtianwen/Chapter_1
%addpath C:/Users/mtianwen/Downloads
addpath '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEGMatlab';

% some global constants:
num_letter = 19;
if K_num == 154 || K_num == 190
    num_repetition = 20;
    odd_seq_len = 10;
    even_seq_len = 10;
else
    % we remove the first odd sequence for training purpose
    num_repetition = 14;
    odd_seq_len = 7;
    even_seq_len = 7;
end
num_rep = 12;
n_length = 25;
num_electrode = 16;
seq_length = n_length * num_electrode;
prop_num = round(0.3 * seq_length);

data_file_name = strcat('K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_eeg_dat_ML_', file_subscript, '_', ...
    int2str(dec_factor), '_from_', eeg_file_subscript);
eeg_dat = load(strcat(data_file_name, '_odd.mat'));
label = eeg_dat.eeg_type';
signal = eeg_dat.eeg_signals;
signal = double(signal);
dir1 = strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num));


%%
mkdir(dir1, '/swLDA')
% Permute the array, reshape the array
signal = permute(signal, [3, 2, 1]);

%%
signal = reshape(signal, [seq_length, num_letter * odd_seq_len * num_rep]);
% label = reshape(label, [num_letter, num_repetition, num_rep]);
% disp(size(label));

mkdir(dir1, strcat('/swLDA/all_channels'));
%%


[b, se, pval, inmodel, stats] = train_SWLDAmatlab(signal.', label, ...
    prop_num);


% save swlda wts
dir2 = strcat('/K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_swlda_wts_train_', file_subscript, ...
    '_', int2str(dec_factor), '_all_channels_', ...
    'odd_', eeg_file_subscript, '_zeta_', num2str(1-zeta_binary_threshold), '.mat');
disp(dir2);

save(strcat(dir1, '/swLDA/all_channels', dir2), ...
    'b', 'se', 'pval', 'inmodel', 'stats');

