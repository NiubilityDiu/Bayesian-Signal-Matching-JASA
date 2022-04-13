rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
local_use = T
# local_use = (args[1] == 'T' || args[1] == 'True')
# library(rstan)
library(mvtnorm)
library(R.matlab)
library(ggplot2)

if (local_use) {
  parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  # parent_dir = 'K:\\Dissertation\\Dataset and Rcode'
  # parent_dir = 'C:/Users/mtwen/Downloads'
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  # rstan_options(auto_write=T)  # only for local running
  # sub_new = 'sub_1'
  # letter_num_new = 4
} else {
  parent_dir = '/home/mtianwen'
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  # rstan_options(auto_write=F)  # only for server running
  # sub_new = K_sub_ids[as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))]
  # letter_num_new = as.integer(args[2])
}
source(file.path(parent_dir_r, 'self_R_fun', 'gibbs_sampling_fun.R'))

total_sub_num = 4
size_length = total_sub_num - 1
seed_num = 612
set.seed(seed_num)
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data')
# data_type_trn = 'TRN_files'
# sub_source = paste('sub_', 2:4, sep='')
sim_name = 'sim_2_case_1'
target_char = 'T'
target_row_col = determine_row_column(target_char, rcp_key_array)
target_row = target_row_col$row
target_col = target_row_col$column

# label_seq = rep(0, 10)
label_seq = sample(c(0,1), size=10, replace=T)
# 0: source; 1: new
label_length = length(label_seq)

param_true_ls = list(
  new = list(mu_ntar=0, mu_tar=0.5, sd=0.5),
  source = list(
    sub_2 = list(mu_ntar=-0.5, mu_tar=0, sd=0.5)
  )
)


new_data_ls = list(
  score = NULL,
  type = NULL,
  code = NULL,
  label = NULL
)

source_data_ls = list(
  sub_2 = list(
    score = NULL,
    type = NULL,
    code = NULL,
    label = NULL
  )
)

for (seq_i in 1:label_length) {
  if (label_seq[seq_i]) {
    # new
    mu_i = rep(param_true_ls$new$mu_ntar, 12)
    mu_i[c(target_row, target_col)] = param_true_ls$new$mu_tar
    cov_i = param_true_ls$new$sd^2 * diag(12)
    
    score_i = as.numeric(rmvnorm(n=1, mean=mu_i, sigma=cov_i))
    output_i = generate_stimulus_group_sequence(target_char, rcp_key_array)
    code_i = output_i$code
    type_i = output_i$type
    score_i_reorder = score_i[code_i]
    new_data_ls$score = cbind(new_data_ls$score, score_i_reorder)
    new_data_ls$type = cbind(new_data_ls$type, type_i)
    new_data_ls$code = cbind(new_data_ls$code, code_i)
    new_data_ls$label = c(new_data_ls$label, seq_i)
  } else {
    # source
    mu_i = rep(param_true_ls$source$sub_2$mu_ntar, 12)
    mu_i[c(target_row, target_col)] = param_true_ls$source$sub_2$mu_tar
    cov_i = param_true_ls$source$sub_2$sd^2 * diag(12)
    
    score_i = as.numeric(rmvnorm(n=1, mean=mu_i, sigma=cov_i))
    output_i = generate_stimulus_group_sequence(target_char, rcp_key_array)
    code_i = output_i$code
    type_i = output_i$type
    score_i_reorder = score_i[code_i]
    source_data_ls$sub_2$score = cbind(source_data_ls$sub_2$score, score_i_reorder)
    source_data_ls$sub_2$type = cbind(source_data_ls$sub_2$type, type_i)
    source_data_ls$sub_2$code = cbind(source_data_ls$sub_2$code, code_i)
    source_data_ls$sub_2$label = c(source_data_ls$sub_2$label, seq_i)
  }
}
