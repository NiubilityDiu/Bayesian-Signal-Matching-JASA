# evaluate the prediction accuracy of single-channel
rm(list=ls(all.names=T))
local_use = T

library(R.matlab)
library(ggplot2)
library(gridExtra)

parent_dir = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
n_comp = 2

parent_dir_r = file.path(parent_dir, 'R')
parent_trn_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'TRN_files')
parent_frt_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'FRT_files')

source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))

seq_size_train = 5
target_char_train_size = 5
select_channel_ids = 1:16
select_channel_size = length(select_channel_ids)

sub_new_name = 'K151'
print(sub_new_name)
new_sub_frt_data_dir = file.path(parent_frt_dir, sub_new_name)

sens_name_vec = NULL
length_vec = c(0.35, 0.3, 0.25)
gamma_vec = c(1.25, 1.2, 1.15)
for (length_iter in length_vec) {
  for (gamma_iter in gamma_vec) {
    sens_name_vec = c(sens_name_vec, 
                      paste('length_', length_iter, '_', length_iter - 0.1, '_gamma_', 
                            gamma_iter, sep=''))
  }
}

# import sensitivity prediction accuracy
df_bkm_mixture_test_sens = readRDS(file.path(new_sub_frt_data_dir, 'Prediction', 'xDAWN_comp_2_reduced', 
                                             'K151_bkm_mixture_test_sensitivity.RDS'))
frt_common_name_vec = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')
seq_size_test = 0
seq_size_test = lapply(
  1:length(frt_common_name_vec), function(x) nrow(df_bkm_mixture_test_sens[[1]][[frt_common_name_vec[x]]]))
seq_size_test = min(unlist(seq_size_test))
predict_iter_test_mean = matrix(0, nrow=seq_size_test, ncol=9, byrow=T)

for (frt_common_name in frt_common_name_vec) {
  for (name_id in 1:9) {
    sens_name_iter = sens_name_vec[name_id]
    predict_iter_test_mean[, name_id] = predict_iter_test_mean[, name_id] +
      df_bkm_mixture_test_sens[[sens_name_iter]][[frt_common_name]][1:seq_size_test, seq_size_train]
  }
}
df_target_char_test_size = readRDS(file.path(new_sub_frt_data_dir, 'Prediction', 'xDAWN_comp_2_reduced', 
                                             'K151_target_char_test_size.RDS'))
target_char_test_sum = sum(df_target_char_test_size)
predict_iter_test_mean = predict_iter_test_mean / target_char_test_sum
print('Table S5:')
print(signif(predict_iter_test_mean * 100, digits=3))
