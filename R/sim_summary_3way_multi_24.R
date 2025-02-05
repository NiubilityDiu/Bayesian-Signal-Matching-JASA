# evaluate the prediction accuracy.
rm(list=ls(all.names=T))
local_use = T

library(R.matlab)
library(ggplot2)
library(gridExtra)


parent_dir = '/Users/niubilitydiu/Desktop/BSM-Code-V2'

N = 24
K = 24
iter_total_num = 100
record_prediction_bool = F
log_lhd_diff_approx_val = '2.0'

n_k_name = paste('N_', N, '_K_', K, sep='')
cluster_name = 'borrow_gibbs'
ref_name = 'reference_numpyro'
mdwm_name = 'MDWM'
mixture_name = 'mixture_gibbs'


parent_dir_r = file.path(parent_dir, 'R')
source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
parent_sim_data_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'SIM_files')
scenario_name_dir = paste(n_k_name, '_multi_xdawn_eeg', sep='')
parent_sim_data_dir = file.path(parent_sim_data_dir, scenario_name_dir)

target_char_train = 5
seq_size_train = 10
seq_size_test = 10
seq_size_vec = 2:5

dir.create(file.path(parent_sim_data_dir, 'prediction_summary'))
parent_sim_data_dir_2 = file.path(parent_sim_data_dir, 'prediction_summary')

sim_true_letter_train_len = 5

### saving accuracy numbers ###
df_bsm_mixture_train = df_bsm_cluster_train = df_bsm_reference_train = 
  df_swLDA_reference_train = df_mdwm_mixture_train = 
  array(-9, dim=c(seq_size_train, length(seq_size_vec), iter_total_num))

df_bsm_mixture_test = df_bsm_cluster_test = df_bsm_reference_test = 
  df_swLDA_reference_test = df_mdwm_mixture_test = 
  array(-9, dim=c(seq_size_test, length(seq_size_vec), iter_total_num))

df_bsm_cluster_z = array(-1, dim=c(N-1, length(seq_size_vec), iter_total_num))

if (record_prediction_bool) {
  for (iter_id in 1:iter_total_num) {
    print(iter_id)
    iter_dir = file.path(parent_sim_data_dir, paste('iter_', iter_id-1, sep=''))
    
    for (seq_order_id in 1:length(seq_size_vec)) {
      seq_train_iter = seq_size_vec[seq_order_id]
      # cluster mcmc
      bsm_output = record_sim_prediction(
        sim_true_letter_train_len, seq_train_iter, sim_true_letter_len, seq_size_test,
        iter_dir, cluster_name, 'cluster'
      )
      df_bsm_cluster_train[1:seq_train_iter, seq_order_id, iter_id] = bsm_output$train
      df_bsm_cluster_test[, seq_order_id, iter_id] = bsm_output$test
      
      bsm_mcmc_iter_seq_id_dir = file.path(iter_dir, cluster_name, 
                                           paste('mcmc_sub_0_seq_size_', seq_train_iter, '_cluster_log_lhd_diff_approx_',
                                                 log_lhd_diff_approx_val, '.mat', sep=''))
      bsm_mcmc_iter_seq_id = readMat(bsm_mcmc_iter_seq_id_dir)
      df_bsm_cluster_z[, seq_order_id, iter_id] = apply(rbind(bsm_mcmc_iter_seq_id$chain.1[[10]], 
                                                              bsm_mcmc_iter_seq_id$chain.2[[10]]), 2, mean)
      # reference mcmc
      bsm_reference_output = record_sim_prediction(
        sim_true_letter_train_len, seq_train_iter, sim_true_letter_len, seq_size_test,
        iter_dir, ref_name, 'reference'
      )
      df_bsm_reference_train[1:seq_train_iter, seq_order_id, iter_id] = bsm_reference_output$train
      df_bsm_reference_test[, seq_order_id, iter_id] = bsm_reference_output$test
      
      # mixture mcmc
      bsm_mixture_output = record_sim_prediction(
        sim_true_letter_train_len, seq_train_iter, sim_true_letter_len, seq_size_test,
        iter_dir, mixture_name, 'mixture'
      )
      df_bsm_mixture_train[1:seq_train_iter, seq_order_id, iter_id] = bsm_mixture_output$train
      df_bsm_mixture_test[, seq_order_id, iter_id] = bsm_mixture_output$test
      
      # swLDA
      swlda_output = record_sim_prediction(
        sim_true_letter_train_len, seq_train_iter, sim_true_letter_len, seq_size_test,
        iter_dir, 'swLDA', 'swLDA'
      )
      df_swLDA_reference_train[1:seq_train_iter, seq_order_id, iter_id] = swlda_output$train
      df_swLDA_reference_test[, seq_order_id, iter_id] = swlda_output$test
      
      # MDWM      
      mdwm_output = record_sim_prediction(
        sim_true_letter_train_len, seq_train_iter, sim_true_letter_len, seq_size_test,
        iter_dir, 'MDWM', 'MDWM'
      )
      df_mdwm_mixture_train[1:seq_train_iter, seq_order_id, iter_id] = mdwm_output$train
      df_mdwm_mixture_test[, seq_order_id, iter_id] = mdwm_output$test
      
    }
  }
  
  saveRDS(df_bsm_mixture_train, file.path(parent_sim_data_dir_2, 'bsm_mixture_train.RDS'))
  saveRDS(df_bsm_cluster_train, file.path(parent_sim_data_dir_2, 'bsm_cluster_train.RDS'))
  saveRDS(df_bsm_reference_train, file.path(parent_sim_data_dir_2, 'bsm_reference_train.RDS'))
  saveRDS(df_swLDA_reference_train, file.path(parent_sim_data_dir_2, 'swLDA_reference_train.RDS'))
  saveRDS(df_mdwm_mixture_train, file.path(parent_sim_data_dir_2, 'MDWM_mixture_train.RDS'))
  
  saveRDS(df_bsm_mixture_test, file.path(parent_sim_data_dir_2, 'bsm_mixture_test.RDS'))
  saveRDS(df_bsm_cluster_test, file.path(parent_sim_data_dir_2, 'bsm_cluster_test.RDS'))
  saveRDS(df_bsm_reference_test, file.path(parent_sim_data_dir_2, 'bsm_reference_test.RDS'))
  saveRDS(df_swLDA_reference_test, file.path(parent_sim_data_dir_2, 'swLDA_reference_test.RDS'))
  saveRDS(df_mdwm_mixture_test, file.path(parent_sim_data_dir_2, 'MDWM_mixture_test.RDS'))
  saveRDS(df_bsm_cluster_z, file.path(parent_sim_data_dir_2, 'bsm_cluster_z.RDS'))
  
} else {
  
  df_bsm_mixture_test = readRDS(file.path(parent_sim_data_dir_2, 'bsm_mixture_test.RDS'))
  df_bsm_cluster_test = readRDS(file.path(parent_sim_data_dir_2, 'bsm_cluster_test.RDS'))
  df_bsm_reference_test = readRDS(file.path(parent_sim_data_dir_2, 'bsm_reference_test.RDS'))
  df_swLDA_reference_test = readRDS(file.path(parent_sim_data_dir_2, 'swLDA_reference_test.RDS'))
  df_mdwm_mixture_test = readRDS(file.path(parent_sim_data_dir_2, 'MDWM_mixture_test.RDS'))
  df_bsm_cluster_z = readRDS(file.path(parent_sim_data_dir_2, 'bsm_cluster_z.RDS'))
  
  # visualize mean of Z across 100 replications
  df_bsm_cluster_z_mean = apply(df_bsm_cluster_z, c(1,2), mean)
  df_bsm_cluster_z_sd = apply(df_bsm_cluster_z, c(1,2), sd)
  
  # plot(1:(N-1), df_bsm_cluster_z_mean[,1])
  df_bsm_cluster_z_long = data.frame(
    z_mean = as.numeric(df_bsm_cluster_z_mean),
    z_sd = as.numeric(df_bsm_cluster_z_sd),
    subject_id = rep(1:(N-1), 4),
    seq_size_train = rep(2:5, each=N-1)
  )
  df_bsm_cluster_z_long$z_low = df_bsm_cluster_z_long$z_mean - df_bsm_cluster_z_long$z_sd
  df_bsm_cluster_z_long$z_upp = df_bsm_cluster_z_long$z_mean + df_bsm_cluster_z_long$z_sd
  df_bsm_cluster_z_long$seq_size_train = factor(
    df_bsm_cluster_z_long$seq_size_train, levels=2:5, 
    labels=paste('Training Sequence Size ', 2:5, sep='')
  )
  p_bsm_cluster_z_mean = ggplot(data=df_bsm_cluster_z_long, 
                                aes(x=subject_id, y=z_mean, group=factor(seq_size_train))) +
    geom_point() + geom_line(linetype=2, alpha=.25) +
    geom_hline(yintercept=0.10, linetype=2) + 
    facet_wrap(~seq_size_train, nrow=2, ncol=2) +
    geom_errorbar(aes(ymin=z_low, max=z_upp), width=0.5) +
    scale_x_continuous(limits=c(0.5, 23.5)) +
    xlab('Source Participant Index') + ylab('') + 
    ggtitle('Posterior Mean of {Z=1}') + ylim(c(0, 0.25)) +
    theme(plot.title=element_text(hjust=0.5, size=10),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        linewidth = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', linewidth=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.2, .2, .2, .2, 'cm'))
  p_bsm_cluster_z_mean
  p_bsm_cluster_z_mean_dir = file.path(
    parent_sim_data_dir_2,
    paste('plot_p_bsm_cluster_z_seq_train_size_2-5_iteration_', iter_total_num, '.png', sep='')
  )
  ggsave(p_bsm_cluster_z_mean_dir, p_bsm_cluster_z_mean, 
         width=300, height=150, units='mm', dpi=300)
  
  accuracy_level_1 = 0.7; accuracy_level_2 = 0.90
  
  ### produce plots ###
  for (seq_order_id in 1:length(seq_size_vec)) {
    seq_train_iter = seq_size_vec[seq_order_id]
    print(seq_train_iter)
    
    predict_iter_test_mean = c(
      apply(df_bsm_mixture_test[,seq_order_id,], 1, mean), 
      apply(df_bsm_cluster_test[,seq_order_id,], 1, mean),
      apply(df_bsm_reference_test[,seq_order_id,], 1, mean),
      apply(df_swLDA_reference_test[,seq_order_id,], 1, mean),
      apply(df_mdwm_mixture_test[,seq_order_id,], 1, mean)
    )
    predict_iter_test_se = c(
      apply(df_bsm_mixture_test[,seq_order_id,], 1, sd), 
      apply(df_bsm_cluster_test[,seq_order_id,], 1, sd),
      apply(df_bsm_reference_test[,seq_order_id,], 1, sd),
      apply(df_swLDA_reference_test[,seq_order_id,], 1, sd),
      apply(df_mdwm_mixture_test[,seq_order_id,], 1, sd)
    )
    predict_iter_test_df = data.frame(
      seq_size = rep(1:seq_size_test, 5),
      mean = predict_iter_test_mean,
      low = pmax(predict_iter_test_mean - predict_iter_test_se, 0),
      upp = pmin(predict_iter_test_mean + predict_iter_test_se, 1),
      method = rep(c('BSM-Mixture', 'BSM', 'BSM-Reference', 'swLDA', 'MDWM'), each=seq_size_test) 
    )
    predict_iter_test_df$method = factor(
      predict_iter_test_df$method, 
      levels=c('BSM-Mixture', 'BSM', 'BSM-Reference', 'MDWM', 'swLDA')
    )
    
    p_predict_test_iter = ggplot(data=predict_iter_test_df, aes(x=seq_size, y=mean)) +
      geom_point() + geom_line() +
      geom_hline(yintercept=accuracy_level_1, linetype=2) + 
      geom_hline(yintercept=accuracy_level_2, linetype=2) +
      facet_wrap(~method, nrow=1, ncol=5) +
      geom_ribbon(aes(ymin=low, max=upp), alpha=.5, fill='grey') +
      scale_x_continuous(breaks=1:seq_size_test) +
      xlab(ifelse(seq_order_id < 10, '', 'Sequence Size of Testing Data')) + 
      ylab('Accuracy') +
      ggtitle(paste('The First ', seq_train_iter, ' Sequence(s) of Training Data', sep='')) + 
      ylim(c(0, 1)) +
      theme(plot.title=element_text(hjust=0.5, size=10),
            panel.background=element_rect(fill = "white",
                                          colour = "black",
                                          linewidth = 0.5, linetype = "solid"),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            legend.position='bottom',
            legend.title=element_blank(),
            legend.background=element_rect(fill='transparent', linewidth=0.2,
                                           color='white', linetype='solid'),
            plot.margin=margin(.2, .2, .2, .2, 'cm'))
    p_predict_iter_test_dir = file.path(
      parent_sim_data_dir_2,
      paste('plot_predict_test_fix_seq_train_size_', seq_train_iter, '_iteration_', iter_total_num, '.png', sep='')
    )
    ggsave(p_predict_iter_test_dir, p_predict_test_iter, width=240, height=60, units='mm', dpi=300)
    
  }
  
  # summarize the prediction accuracy in a table
  df_test_method_list = list(
    'BSM-Mix' = df_bsm_mixture_test,
    'BSM-Cluster' = df_bsm_cluster_test,
    'MDWM' = df_mdwm_mixture_test,
    'BSM-Ref' = df_bsm_reference_test,
    'swLDA' = df_swLDA_reference_test
  )
  
  export_test_percent_table(df_test_method_list, 0.80, 2:5, 2:10, parent_sim_data_dir_2)
  export_test_percent_table(df_test_method_list, 0.85, 2:5, 2:10, parent_sim_data_dir_2)
  export_test_percent_table(df_test_method_list, 0.90, 2:5, 2:10, parent_sim_data_dir_2)
  
}





