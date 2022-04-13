rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
# local_use = T
local_use = (args[1] == 'T' || args[1] == 'True')
# library(rstan)
library(rjson)
library(R.matlab)
library(ggplot2)
library(gridExtra)

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

# here, match pattern refers to the source subject ids.
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
predict_prob_low_mat = array(-1, dim=c(N-1, seq_size_len, iter_total_num))
predict_prob_upp_mat = array(-1, dim=c(N-1, seq_size_len, iter_total_num))


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
      predict_prob_low_mat[sub_iter, seq_iter_id, iter_id] = quantile(prob_iter_mcmc, 0.05)
      predict_prob_upp_mat[sub_iter, seq_iter_id, iter_id] = quantile(prob_iter_mcmc, 0.95)
    }
  }
}

# Find out the subset
subset_indices = apply(sapply(1:iter_total_num, function(x) true_prob_vec[,x] == match_pattern), 2, prod)

# predict_prob_mat_sub = predict_prob_mat[, , subset_indices==1]
predict_prob_sub_mean = apply(predict_prob_mean_mat[,,subset_indices==1], c(1, 2), mean)
predict_prob_sub_low = apply(predict_prob_low_mat[,,subset_indices==1], c(1, 2), mean)
predict_prob_sub_upp = apply(predict_prob_upp_mat[,,subset_indices==1], c(1, 2), mean)

p_predict_prob_ls = lapply(1:(N-1), function(x) NULL)
names(p_predict_prob_ls) = paste('Subject ', 1:(N-1), sep='')
for (sub_iter in 1:(N-1)) {
  predict_prob_sub_df = data.frame(
    seq_size = seq_size_vec,
    mean = predict_prob_sub_mean[sub_iter, ],
    low = predict_prob_sub_low[sub_iter, ],
    upp = predict_prob_sub_upp[sub_iter, ]
  )
  p_iter_name = paste('Subject ', sub_iter, sep='')
  p_predict_prob_ls[[p_iter_name]] = ggplot(data=predict_prob_sub_df, aes(x=seq_size, y=mean)) +
    geom_point() + geom_line() + 
    geom_ribbon(aes(ymin=low, max=upp), alpha=.2, fill='blue') +
    scale_x_continuous(breaks=1:10) +
    xlab('Sequence Size of Subject 0') + 
    ylab(paste('Prob Belonging to Cluster ', sim_dat$subject_0$label, sep='')) + 
    ggtitle(paste(p_iter_name, ', True Label: ', match_pattern[sub_iter], sep='')) + ylim(c(0, 1)) +
    theme(plot.title=element_text(hjust=0.5, size=10),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.2, .2, .2, .2, 'cm'))
}

p_predict_prob_merge = marrangeGrob(p_predict_prob_ls, nrow=2, ncol=4, top=scenario_name_dir)
p_predict_prob_sub_dir = file.path(
  parent_sim_data_dir, 
  paste('plot_prob_cluster_0_iteration_', iter_total_num, '.png', sep='')
)
ggsave(p_predict_prob_sub_dir, p_predict_prob_merge, width=300, height=150, units='mm', dpi=300)

# save by each subject as well
for (sub_iter in 1:(N-1)) {
  p_predict_prob_sub_iter_dir = file.path(
    parent_sim_data_dir, 
    paste('plot_prob_cluster_0_iteration_', iter_total_num, '_subject_', sub_iter, '.png', sep='')
  )
  ggsave(p_predict_prob_sub_iter_dir, p_predict_prob_ls[[sub_iter]], 
         width=150, height=150, units='mm', dpi=300)
}




# binary_output = list(
#   total_combine = total_combine_iter, se_combine = se_combine_iter, sp_combine = sp_combine_iter,
#   total_new = total_new_iter, se_new = se_new_iter, sp_new = sp_new_iter
# )
# 
# # save the mcmc samples
# binary_output_dir = file.path(
#   parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''),
#   paste('sim_source_4_new_seq_', seq_i, '_binary_output_threshold_', phi_mean_threshold, '.rds', sep='')
# )
# saveRDS(binary_output, binary_output_dir)
# 
# # produce the boxplot
# binary_output_df = data.frame(
#   total = c(binary_output$total_combine, binary_output$total_new),
#   se = c(binary_output$se_combine, binary_output$se_new),
#   sp = c(binary_output$sp_combine, binary_output$sp_new),
#   type = rep(c('Combine', 'New Only'), each=rep_num)
# )
# p_binary_total = ggplot(data=binary_output_df, aes(x=type, y=total, group=type)) + 
#   geom_boxplot() + xlab('') + ylab('') + ggtitle('Total') + ylim(c(0.5, 1)) +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.2, .2, .2, .2, 'cm'))
# p_binary_se = ggplot(data=binary_output_df, aes(x=type, y=se, group=type)) + 
#   geom_boxplot() + xlab('') + ylab('') + ggtitle('Sensitivity') + ylim(c(0.5, 1)) +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.2, .2, .2, .2, 'cm'))
# p_binary_sp = ggplot(data=binary_output_df, aes(x=type, y=sp, group=type)) + 
#   geom_boxplot() + xlab('') + ylab('') + ggtitle('Specificity') + ylim(c(0.5, 1)) +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.2, .2, .2, .2, 'cm'))
# 
# p_binary_3 = grid.arrange(
#   p_binary_total, p_binary_se, p_binary_sp, nrow=1, ncol=3, 
#   top=paste('Seq Size ', seq_i, ' Threshold ', phi_mean_threshold, sep='')
# )
# 
# # save the mcmc samples
# binary_plot_dir = file.path(
#   parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''),
#   paste('sim_source_4_new_seq_', seq_i, '_binary_output_threshold_', phi_mean_threshold, '.png', sep='')
# )
# ggsave(binary_plot_dir, p_binary_3, width=200, height=100, units='mm', dpi=300)
