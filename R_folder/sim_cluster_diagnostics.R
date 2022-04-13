# evaluate the sensitivity, specificty, ROC curve on the clustering level with semi-supervised setting.
rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
# local_use = T
local_use = (args[1] == 'T' || args[1] == 'True')
# library(rstan)
library(rjson)
library(R.matlab)
library(ggplot2)
library(gridExtra)
library(pROC)

if (local_use) {
  parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  # parent_dir = 'K:\\Dissertation\\Dataset and Rcode'
  # parent_dir = 'C:/Users/mtwen/Downloads'
  N = 9
  K = 4
  option_id = 9
  sigma_val = as.character("3.0")
  rho_val = as.character("0.5")
  iter_total_num = 10
} else {
  parent_dir = '/home/mtianwen'
  N = as.integer(args[2])
  K = as.integer(args[3])
  option_id = as.integer(args[4])
  sigma_val = args[5]
  rho_val = args[6]
  iter_total_num = 100
}
n_k_name = paste('N_', N, '_K_', K, sep='')

parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
parent_sim_data_dir = file.path(
  parent_dir, 'EEG_MATLAB_data', 'SIM_files', 'Chapter_3', 'numpyro_output', n_k_name
)
scenario_name_dir = paste(
  n_k_name, '_K114_based_option_', option_id, '_sigma_', sigma_val, '_rho_', rho_val, sep=''
)
parent_sim_data_dir = file.path(
  parent_sim_data_dir, scenario_name_dir
)

if (K == 2) {
  if (option_id == 0) {
    match_pattern = c(1, 1, 1)
  } else if (option_id == 1) {
    match_pattern = c(0, 1, 1)
  } else if (option_id == 2) {
    match_pattern = c(0, 0, 1)
  } else {
    match_pattern = c(0, 0, 0)
  }
} else if (K == 3) {
  if (option_id == 0) {
    match_pattern = c(1, 1, 2)
  } else if (option_id == 1) {
    match_pattern = c(1, 2, 2)
  } else if (option_id == 2) {
    match_pattern = c(0, 1, 2)
  } else if (option_id == 3) {
    match_pattern = c(0, 0, 1)
  } else if (option_id == 4) {
    match_pattern = c(0, 0, 2)
  } else if (option_id == 5) {
    match_pattern = c(0, 0, 0)
  } else if (option_id == 6) {
    match_pattern = c(0, 1, 2)
  } else if (option_id == 7) {
    match_pattern = c(0, 1, 2)
  } else {
    match_pattern = c(0, 1, 2)
  }
} else {
  match_pattern = rep(0:3, each=2)
}

true_prob_vec = matrix(-9, nrow=N-1, ncol=iter_total_num)
seq_size_vec = 1:10
seq_size_len = length(seq_size_vec)
predict_prob_mean_mat = array(-1, dim=c(N-1, seq_size_len, iter_total_num))

for (iter_id in 1:iter_total_num) {
  iter_dir = file.path(parent_sim_data_dir, paste('iter_', iter_id-1, sep=''))
  
  # import data files
  sim_name= 'sim_dat'
  data_dir = file.path(iter_dir, paste(sim_name, '.json', sep=''))
  print(data_dir)
  sim_dat = fromJSON(file=data_dir)
  for (sub_iter in 1:(N-1)) {
    sub_iter_name = paste('subject_', sub_iter, sep='')
    true_prob_vec[sub_iter, iter_id] = sim_dat[[sub_iter_name]]$label
  }
  
  # import mcmc samples
  for (seq_iter_id in 1:seq_size_len) {
    seq_iter = seq_size_vec[seq_iter_id]
    mcmc_seq_iter_dir = file.path(iter_dir, paste('*mcmc_seq_size_', seq_iter, '.mat', sep=''))
    mcmc_seq_iter_mat = readMat(
      Sys.glob(mcmc_seq_iter_dir)
    )
    for (sub_iter in 1:(N-1)) {
      prob_iter_name = paste('prob', sub_iter, sep='.')
      prob_iter_mcmc = mcmc_seq_iter_mat[[prob_iter_name]][, 1]
      predict_prob_mean_mat[sub_iter, seq_iter_id, iter_id] = mean(prob_iter_mcmc)
    }
  }
}


# show with respect to sub_id
# It may not work well for K=3 because some cases do not have controls or cases.

labels_general = rep('Not Match', K)
labels_general[sim_dat$subject_0$label+1] = 'Match'
if (K == 2) {
  pdf(file.path(parent_sim_data_dir,
                paste('plot_ROC_iteration_', iter_total_num, '_per_subject_iteration.pdf', sep='')))
  for (sub_id in 1:(N-1)) {
    for (seq_id in 1:seq_size_len) {
      pROC_obj_sub_seq_iter = roc(
        factor(true_prob_vec[sub_id,], levels=0:(K-1), labels=labels_general), 
        predict_prob_mean_mat[sub_id, seq_id,])
      auc_sub_seq_iter = signif(pROC_obj_sub_seq_iter$auc, digits=2)
      plot(pROC_obj_sub_seq_iter, 
           main=paste('Source Subject ', sub_id, ', Seq Size=', seq_id, ', AUC=', 
                      auc_sub_seq_iter, sep=""))
    }
  }
  dev.off()
}

# collapse with respect to sub_id
pdf(file.path(parent_sim_data_dir, 
              paste('plot_ROC_iteration_', iter_total_num, '_per_iteration.pdf', sep='')))
for (seq_id in 1:seq_size_len) {
  pROC_obj_seq_iter = roc(
    factor(as.numeric(true_prob_vec), levels=0:(K-1), labels=labels_general), 
    as.numeric(predict_prob_mean_mat[, seq_id,]))
  auc_seq_iter = signif(pROC_obj_seq_iter$auc, digits=2)
  plot(pROC_obj_seq_iter, main=paste('Seq Size=', seq_id, ', AUC=', auc_seq_iter, sep=""))
}
dev.off()
