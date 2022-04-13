function [eeg_cov_obj] = constructCovMatRCP(eeg_trunc_obj, template_tar, template_ntar)
%constructCovMatRCP Summary of this function goes here
%   Detailed explanation goes here
% Construct template P1 signal matrices

eeg_trunc_mat = eeg_trunc_obj.Signal;
eeg_type = eeg_trunc_obj.Type;
[flash_num, ~, channel_num] = size(eeg_trunc_mat);

if isempty(template_tar)
    template_tar = squeeze(mean(eeg_trunc_mat(eeg_type == 1, :, :), 1));
end

if isempty(template_ntar)
    template_ntar = squeeze(mean(eeg_trunc_mat(eeg_type == -1, :, :), 1));
end

cov_feature_flat = zeros(flash_num, (1+2*channel_num)*channel_num);
for i=1:flash_num
    if eeg_type(i, :) == 1
        x_aug_i = cat(2, squeeze(eeg_trunc_mat(i, :, :)), template_tar);
    else
        x_aug_i = cat(2, squeeze(eeg_trunc_mat(i, :, :)), template_ntar);
    end
    x_aug_i_triu = triu(cov(x_aug_i));
    cov_feature_flat(i, :) = convertTriUpperVector(x_aug_i_triu);
end
eeg_cov_obj = struct();
eeg_cov_obj.Signal = cov_feature_flat;
eeg_cov_obj.Type = eeg_type;
eeg_cov_obj.Code = eeg_trunc_obj.Code;
eeg_cov_obj.IndexTag = eeg_trunc_obj.IndexTag;
eeg_cov_obj.IndexBegin = eeg_trunc_obj.IndexBegin;
eeg_cov_obj.Text = eeg_trunc_obj.Text;
eeg_cov_obj.LetterTable = eeg_trunc_obj.LetterTable;
end