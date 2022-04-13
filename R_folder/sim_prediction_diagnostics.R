# evaluate the prediction accuracy.
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
  N = 4
  K = 3
  option_id = 0
  sigma_val = as.character("5.0")
  rho_val = as.character("0.5")
  iter_total_num = 15
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
} else {
  if (option_id == 0) {
    match_pattern = c(1, 1, 2)
  } else if (option_id == 1) {
    match_pattern = c(1, 2, 2)
  } else if (option_id == 2) {
    match_pattern = c(0, 1, 2)
  } else if (option_id == 3){
    match_pattern = c(0, 0, 1)
  } else if (option_id == 4){
    match_pattern = c(0, 0, 2)
  } else if (option_id == 5){
    match_pattern = c(0, 0, 0)
  } else if (option_id == 6) {
    match_pattern = c(0, 1, 2)
  } else if (option_id == 7) {
    match_pattern = c(0, 1, 2)
  } else {
    match_pattern = c(0, 1, 2)
  }
}

seq_size_train = seq_size_test = 10
decision_rule_vec = c('NewOnly', 'Strict', 'Flexible')
prob_threshold = 0.5

predict_accuracy_train = array(0, dim=c(seq_size_train, 3, seq_size_train, iter_total_num))
for (iter_id in 1:iter_total_num) {
  iter_dir = file.path(parent_sim_data_dir, paste('iter_', iter_id-1, sep=''), 'swLDA')
  
  for (seq_train_iter in 1:seq_size_train) {
    # import json file.
    swlda_prob_name = paste('swLDA_predict_train_seq_size_', seq_train_iter, '_threshold_', 
                            prob_threshold, sep='')
    swlda_prob_dir = file.path(iter_dir, paste(swlda_prob_name, '.json', sep=''))
    print(swlda_prob_dir)
    swlda_prob_dat = fromJSON(file=swlda_prob_dir)
    for (rule_iter in 1:3) {
      swlda_prob_rule = swlda_prob_dat[[decision_rule_vec[rule_iter]]]
      predict_accuracy_train[1:seq_train_iter, rule_iter, seq_train_iter, iter_id] = 
        swlda_prob_rule[[1]]
    }
  }
}

# look at the prediction accuracy each testing sequence size with respect to training sample size stratified on 
# different decision rules.
for (seq_train_iter in 1:seq_size_train) {
  print(seq_train_iter)
  predict_train_iter_mean = apply(predict_accuracy_train[,,seq_train_iter,], c(1,2), mean)[1:seq_train_iter,]
  predict_train_iter_se = apply(predict_accuracy_train[,,seq_train_iter,], c(1,2), sd)[1:seq_train_iter,]
  predict_train_iter_df = data.frame(
    seq_size = rep(1:seq_train_iter, 3),
    mean = as.vector(predict_train_iter_mean),
    low = pmax(as.vector(predict_train_iter_mean - predict_train_iter_se), 0),
    upp = pmin(as.vector(predict_train_iter_mean + predict_train_iter_se), 1),
    rule = rep(decision_rule_vec, each=seq_train_iter)
  )
  predict_train_iter_df$rule = factor(predict_train_iter_df$rule, levels=c('NewOnly', 'Strict', 'Flexible'))
  p_predict_train_iter = ggplot(data=predict_train_iter_df, aes(x=seq_size, y=mean)) +
    geom_point() + geom_line() + 
    geom_hline(yintercept=0.5, linetype=2) + geom_hline(yintercept=0.9, linetype=2) +
    facet_wrap(~rule, nrow=1, ncol=3) + 
    geom_ribbon(aes(ymin=low, max=upp), alpha=.5, fill='grey') +
    scale_x_continuous(breaks=1:seq_train_iter) +
    xlab('Sequence Size of Training Set') + ylab('Subject 0') + 
    ggtitle(paste('Probability of T of Training Data at Seq ', seq_train_iter, sep='')) + ylim(c(0, 1)) +
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
  
  p_predict_train_iter_dir = file.path(
    parent_sim_data_dir, 
    paste('plot_prob_predict_iteration_', iter_total_num, '_seq_train_size_', seq_train_iter, '.png', sep='')
  )
  ggsave(p_predict_train_iter_dir, p_predict_train_iter, width=300, height=150, units='mm', dpi=300)
}





### testing only ###
# predict_accuracy_test = array(-9, dim=c(seq_size_test, 3, seq_size_train, iter_total_num))
# for (iter_id in 1:iter_total_num) {
#   iter_dir = file.path(parent_sim_data_dir, paste('iter_', iter_id-1, sep=''), 'swLDA')
#   
#   for (seq_train_iter in 1:seq_size_train) {
#     # import json file.
#     swlda_predict_name = paste('swLDA_predict_seq_size_', seq_train_iter, '_threshold_', 
#                                prob_threshold, sep='')
#     swlda_predict_dir = file.path(iter_dir, paste(swlda_predict_name, '.json', sep=''))
#     print(swlda_predict_dir)
#     swlda_predict_dat = fromJSON(file=swlda_predict_dir)
#     
#     for (rule_iter in 1:3) {
#       swlda_predict_rule = swlda_predict_dat[[decision_rule_vec[rule_iter]]]
#       for (seq_test_iter in 1:seq_size_test) {
#         seq_test_iter_name = paste('seq_test_', seq_test_iter, sep='')
#         predict_accuracy_test[seq_test_iter, rule_iter, seq_train_iter, iter_id] = 
#           swlda_predict_rule[[seq_test_iter_name]][['letter']]
#       }
#     }
#   }
# }
# 
# # Marginal comparison first
# # 1. training sequence size
# predict_marginal_train_seq_mean = apply(predict_accuracy_test, 3, mean)
# predict_marginal_train_seq_se = apply(predict_accuracy_test, 3, sd)
# predict_marginal_train_seq_df = data.frame(
#   seq_size = 1:seq_size_train,
#   mean = predict_marginal_train_seq_mean,
#   low = pmax(predict_marginal_train_seq_mean - predict_marginal_train_seq_se, 0),
#   upp = pmin(predict_marginal_train_seq_mean + predict_marginal_train_seq_se, 1)
# )
# p_predict_marginal_train_seq = ggplot(data=predict_marginal_train_seq_df, aes(x=seq_size, y=mean)) +
#   geom_point() + geom_line() + 
#   geom_ribbon(aes(ymin=low, max=upp), alpha=.2, fill='blue') +
#   scale_x_continuous(breaks=1:10) +
#   xlab('Sequence Size of Subject 0 (Training)') + ylab('') + 
#   ggtitle('Prediction Accuracy of Subject 0 (Testing)') + ylim(c(0, 1)) +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         # legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.2, .2, .2, .2, 'cm'))
# p_predict_marginal_train_seq
# 
# # 2. decision rules
# predict_marginal_decision_mean = apply(predict_accuracy_test, 2, mean)
# predict_marginal_decision_se = apply(predict_accuracy_test, 2, sd)
# predict_marginal_decision_df = data.frame(
#   seq_size = decision_rule_vec,
#   mean = predict_marginal_decision_mean,
#   low = pmax(predict_marginal_decision_mean - predict_marginal_decision_se, 0),
#   upp = pmin(predict_marginal_decision_mean + predict_marginal_decision_se, 1)
# )
# predict_marginal_decision_df
# 
# # 3. testing sequence size
# predict_marginal_test_seq_mean = apply(predict_accuracy_test, 1, mean)
# predict_marginal_test_seq_se = apply(predict_accuracy_test, 1, sd)
# predict_marginal_test_seq_df = data.frame(
#   seq_size = 1:seq_size_test,
#   mean = predict_marginal_test_seq_mean,
#   low = pmax(predict_marginal_test_seq_mean - predict_marginal_test_seq_se, 0),
#   upp = pmin(predict_marginal_test_seq_mean + predict_marginal_test_seq_se, 1)
# )
# p_predict_marginal_test_seq = ggplot(data=predict_marginal_test_seq_df, aes(x=seq_size, y=mean)) +
#   geom_point() + geom_line() + 
#   geom_ribbon(aes(ymin=low, max=upp), alpha=.2, fill='blue') +
#   scale_x_continuous(breaks=1:10) +
#   xlab('Sequence Size of Subject 0 (Testing)') + ylab('') + 
#   ggtitle('Prediction Accuracy of Subject 0 (Testing)') + ylim(c(0, 1)) +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         # legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.2, .2, .2, .2, 'cm'))
# p_predict_marginal_test_seq
# 
# 
# # look at the prediction accuracy each testing sequence size with respect to training sample size stratified on 
# # different decision rules.
# for (seq_test_iter in 1:seq_size_test) {
#   predict_test_iter_mean = apply(predict_accuracy_test[seq_test_iter,,,], c(1,2), mean)
#   predict_test_iter_se = apply(predict_accuracy_test[seq_test_iter,,,], c(1,2), sd)
#   predict_test_iter_df = data.frame(
#     seq_size = rep(1:seq_size_train, each=3),
#     mean = as.vector(predict_test_iter_mean),
#     low = pmax(as.vector(predict_test_iter_mean - predict_test_iter_se), 0),
#     upp = pmin(as.vector(predict_test_iter_mean + predict_test_iter_se), 1),
#     rule = rep(decision_rule_vec, seq_size_train)
#   )
#   predict_test_iter_df$rule = factor(predict_test_iter_df$rule, levels=c('NewOnly', 'Strict', 'Flexible'))
#   p_predict_test_iter = ggplot(data=predict_test_iter_df, aes(x=seq_size, y=mean)) +
#     geom_point() + geom_line() + 
#     geom_hline(yintercept=0.5, linetype=2) + geom_hline(yintercept=0.8, linetype=2) +
#     facet_wrap(~rule, nrow=1, ncol=3) + 
#     geom_ribbon(aes(ymin=low, max=upp), alpha=.5, fill='grey') +
#     scale_x_continuous(breaks=1:10) +
#     xlab('Sequence Size of Training Set') + ylab('Subject 0') + 
#     ggtitle(paste('Prediction Accuracy of Testing Set at Seq ', seq_test_iter, sep='')) + ylim(c(0, 1)) +
#     theme(plot.title=element_text(hjust=0.5, size=10),
#           panel.background=element_rect(fill = "white",
#                                         colour = "black",
#                                         size = 0.5, linetype = "solid"),
#           panel.grid.major=element_blank(),
#           panel.grid.minor=element_blank(),
#           legend.position='bottom',
#           legend.title=element_blank(),
#           legend.background=element_rect(fill='transparent', size=0.2,
#                                          color='white', linetype='solid'),
#           plot.margin=margin(.2, .2, .2, .2, 'cm'))
#   # print(p_predict_test_iter)
#   p_predict_test_iter_dir = file.path(
#     parent_sim_data_dir, 
#     paste('plot_prob_predict_iteration_', iter_total_num, '_seq_test_size_', seq_test_iter, '.png', sep='')
#   )
#   ggsave(p_predict_test_iter_dir, p_predict_test_iter, width=300, height=150, units='mm', dpi=300)
# }
