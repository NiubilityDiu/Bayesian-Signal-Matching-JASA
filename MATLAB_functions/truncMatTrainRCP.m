function [eeg_trunc_obj] = truncMatTrainRCP(eeg_obj, eeg_window)
%truncMatTrain Summary of this function goes here
%   Detailed explanation goes here
% eeg_trun_mat has shape (flash_num, eeg_window * channel_num);
% eeg_trun_type has shape (flash_num, 1), and it takes values from {-1,1}.


eeg_signal = eeg_obj.Signal;
% eeg_tags = eeg_obj.Tags;
eeg_code = eeg_obj.Code;
eeg_type = eeg_obj.Type;
% eeg_stim_grp_num = eeg_obj.GroupNum;
eeg_begin = eeg_obj.Begin;
text = eeg_obj.Text;
letter_table = eeg_obj.LetterTable;

[~, channel_num] = size(eeg_signal);

index_tag = findStimStartOffline(eeg_code);
index_begin = findStimStartOffline(eeg_begin);

[flash_num, ~] = size(index_tag);

eeg_trunc_mat = zeros(flash_num, eeg_window * channel_num);
eeg_trunc_type = zeros(flash_num, 1) - 1;
eeg_trunc_code = zeros(flash_num, 1);
% eeg_trunc_tags = zeros(flash_num, hidden_mat_size);
% eeg_trunc_stim_grp_num = zeros(flash_num, 1);

for flash_id = 1:flash_num
    start_id = index_tag(flash_id, :);
    eeg_trunc_mat(flash_id, :) = reshape(eeg_signal(start_id:(start_id+eeg_window-1),:), [], 1);
    % eeg_trunc_tags(flash_id, :) = eeg_tags(start_id, :); 
    eeg_trunc_code(flash_id, :) = eeg_code(start_id, :);
    if eeg_type(start_id, :) == 1
        eeg_trunc_type(flash_id, :) = 1;
    end
    % eeg_trunc_stim_grp_num(flash_id, :) = eeg_stim_grp_num(start_id,:);
end 

eeg_trunc_obj = struct();
eeg_trunc_obj.Signal = eeg_trunc_mat;
eeg_trunc_obj.Type = eeg_trunc_type;
eeg_trunc_obj.Code = eeg_trunc_code;
% eeg_trunc_obj.GroupNum = eeg_trunc_stim_grp_num;
eeg_trunc_obj.IndexTag = index_tag;
eeg_trunc_obj.IndexBegin = index_begin;
eeg_trunc_obj.Text = text;
eeg_trunc_obj.LetterTable = letter_table;
end

