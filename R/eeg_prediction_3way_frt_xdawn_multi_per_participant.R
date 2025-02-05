# evaluate the prediction accuracy of single-channel
rm(list=ls(all.names=T))
local_use = T


library(R.matlab)
library(ggplot2)
library(gridExtra)


parent_dir = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
collect_data_bool = T
parent_dir_r = file.path(parent_dir, 'R')
parent_trn_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'TRN_files')
parent_frt_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'FRT_files')

source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))

cluster_name = 'borrow_gibbs_letter_5_reduced_xdawn'
ref_name = 'reference_numpyro_letter_5_xdawn'

seq_size_vec = 5
target_char_train_size = 5

select_channel_ids = 1:16
select_channel_size = length(select_channel_ids)
if (select_channel_size == 16) {
  channel_name_2 = 'channel_all'
} else {
  channel_name_2 = 'channel_6_15'
}
channel_name_2 = 'channel_all_comp_2'
kernel_name = 'length_0.3_0.2_gamma_1.2'

log_lhd_approx = '1.0'


sub_new_name = 'K151'
print(sub_new_name)
new_sub_frt_data_dir = file.path(parent_frt_dir, sub_new_name)
new_sub_frt_predict_summary_dir = file.path(new_sub_frt_data_dir, 
                                            'Prediction', 'xDAWN_comp_2_reduced')
frt_common_name_vec = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')

# BSM-Mixture
df_bkm_mixture_test_dir = file.path(new_sub_frt_predict_summary_dir, 
                                    paste(sub_new_name, '_bkm_cluster_test.RDS', sep=''))
df_bkm_mixture_test = readRDS(df_bkm_mixture_test_dir)
# MDWM
df_mdwm_test_dir = file.path(new_sub_frt_predict_summary_dir, 
                             paste(sub_new_name, '_df_mdwm_test.RDS', sep=''))
df_mdwm_test = readRDS(df_mdwm_test_dir)
# swLDA
df_swLDA_reference_test_dir = file.path(new_sub_frt_predict_summary_dir, 
                                        paste(sub_new_name, '_swLDA_reference_test.RDS', sep=''))
df_swLDA_reference_test = readRDS(df_swLDA_reference_test_dir)
# SMGP
df_smgp_test_dir = file.path(new_sub_frt_predict_summary_dir, 
                             paste(sub_new_name, '_df_smgp_test.RDS', sep=''))
df_smgp_test = readRDS(df_smgp_test_dir)
# actual spelling key
target_char_test_size_dir = file.path(new_sub_frt_predict_summary_dir, 
                                      paste(sub_new_name, '_target_char_test_size.RDS', sep=''))
target_char_test_size = readRDS(target_char_test_size_dir)
# overall accuracy
target_char_test_size_overall = sum(target_char_test_size)

seq_size_test = lapply(
  1:length(frt_common_name_vec), function(x) nrow(df_bkm_mixture_test[[frt_common_name_vec[x]]]))
seq_size_test = min(unlist(seq_size_test))

df_bkm_mixture_test_accuracy = calculate_test_accuracy_eeg(
  df_bkm_mixture_test, seq_size_vec, 1:seq_size_test, 
  frt_common_name_vec, target_char_test_size_overall
)
df_mdwm_test_accuracy = calculate_test_accuracy_eeg(
  df_mdwm_test, seq_size_vec, 1:seq_size_test, 
  frt_common_name_vec, target_char_test_size_overall
)
df_swLDA_reference_test_accuracy = calculate_test_accuracy_eeg(
  df_swLDA_reference_test, seq_size_vec, 1:seq_size_test, 
  frt_common_name_vec, target_char_test_size_overall
)
df_smgp_reference_test_accuracy = calculate_test_accuracy_eeg(
  df_smgp_test, 1:seq_size_vec, 1:seq_size_test, 
  frt_common_name_vec, target_char_test_size_overall
)[,seq_size_vec]

table_2_df = cbind.data.frame(BSM_Mixture=df_bkm_mixture_test_accuracy, 
                              MDWM=df_mdwm_test_accuracy,
                              swLDA=df_swLDA_reference_test_accuracy, 
                              SMGP=df_smgp_reference_test_accuracy)
# table_2_df
colnames(table_2_df) = c('BSM_Mixture', 'MDWM', 'swLDA', 'SMGP')
print(table_2_df)
