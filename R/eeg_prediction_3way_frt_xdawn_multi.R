# evaluate prediction accuracy of real data analysis
# offline
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

sub_name_vec = eeg_reduced_9_ids
N = length(sub_name_vec)

cluster_name = 'borrow_gibbs_letter_5_reduced'
cluster_xdawn_name = 'borrow_gibbs_letter_5_reduced_xdawn'
prediction_name_2 = paste('xDAWN_comp_', n_comp, '_reduced', sep='')

log_lhd_approx = "1.0"
ref_name = 'reference_numpyro_letter_5'
ref_xdawn_name = 'reference_numpyro_letter_5_xdawn'
ref_xdawn_lda_name = 'reference_numpyro_letter_5_xdawn_lda'
mixture_name = 'mixture_gibbs_letter_5_reduced_xdawn'
mixture_threshold = 0.1
smgp_name = 'BayesGenq2Pred'
parent_smgp_test_dir = file.path(parent_dir, 'EEG_MATLAB_data', smgp_name)

# seq_size_train = 5
seq_size_vec = c(5)
target_char_train_size = 5
select_channel_ids = 1:16
select_channel_size = length(select_channel_ids)
channel_name = 'channel_6_12_14_15_16'
channel_name_2 = paste('channel_all_comp_', n_comp, sep='')
channel_name_3 = paste(channel_name_2, '_mutual_top_10', sep='')
kernel_name = 'length_0.3_0.2_gamma_1.2'

  
### plot frt files ###
parent_frt_predict_dir = file.path(parent_frt_dir, 'Prediction')
df_bkm_mixture_test_all = readRDS(file.path(parent_frt_predict_dir, prediction_name_2, 'K_bkm_mixture_frt.RDS'))
df_mdwm_test_all = readRDS(file.path(parent_frt_predict_dir, prediction_name_2, 'K_mdwm_frt.RDS'))
df_swLDA_reference_test_all = readRDS(file.path(parent_frt_predict_dir, prediction_name_2, 'K_swLDA_reference_frt.RDS'))
df_smgp_test_all = readRDS(file.path(parent_frt_predict_dir, prediction_name_2, 'K_smgp_frt.RDS'))
target_char_test_size_all = readRDS(file.path(parent_frt_predict_dir, prediction_name_2, 'K_target_char_test_size.RDS'))


for (seq_order_id in 1:length(seq_size_vec)) {
  
  seq_train_iter = seq_size_vec[seq_order_id]
  print(seq_train_iter)
  
  # absolute measurement
  p_predict_test_iter_ls = lapply(1:N, function(x) NULL)
  names(p_predict_test_iter_ls) = sub_name_vec
  
  for (sub_new_name in sub_name_vec) {
    print(sub_new_name)
    
    seq_size_test = 0
    if (sub_new_name %in% names(FRT_file_name_ls)) {
      frt_common_name_vec = FRT_file_name_ls[[sub_new_name]]
    } else {
      frt_common_name_vec = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')
    }
    
    # K118 has different seq_size_test across FRT files
    seq_size_test = lapply(
      1:length(frt_common_name_vec), function(x) nrow(df_bkm_mixture_test_all[[sub_new_name]][[frt_common_name_vec[x]]]))
    seq_size_test = min(unlist(seq_size_test))
    
    predict_iter_test_mean = matrix(0, nrow=seq_size_test, ncol=4, byrow=T)
    for (frt_common_name in frt_common_name_vec) {
      
      predict_iter_test_mean[, 1] = predict_iter_test_mean[, 1] +
        df_bkm_mixture_test_all[[sub_new_name]][[frt_common_name]][1:seq_size_test, seq_order_id]
      
      predict_iter_test_mean[, 2] = predict_iter_test_mean[, 2] +
        df_mdwm_test_all[[sub_new_name]][[frt_common_name]][1:seq_size_test, seq_order_id]

      predict_iter_test_mean[, 3] = predict_iter_test_mean[, 3] +
        df_swLDA_reference_test_all[[sub_new_name]][[frt_common_name]][1:seq_size_test, seq_order_id]

      predict_iter_test_mean[, 4] = predict_iter_test_mean[, 4] +
        df_smgp_test_all[[sub_new_name]][[frt_common_name]][1:seq_size_test, seq_order_id]
      
    }
    
    predict_iter_test_mean = predict_iter_test_mean / sum(target_char_test_size_all[[sub_new_name]])
    
    # absolute measurement
    predict_iter_test_df = data.frame(
      seq_size = rep(1:seq_size_test, 4),
      mean = as.vector(predict_iter_test_mean),
      method = rep(c('BSM-Mixture', 'MDWM', 'swLDA-Reference', 'SMGP'),
                   each=seq_size_test)
    )
    predict_iter_test_df$method = factor(
      predict_iter_test_df$method,
      levels=c('BSM-Mixture', 'MDWM', 'swLDA-Reference', 'SMGP')
    )

    p_predict_test_iter = ggplot(data=predict_iter_test_df, aes(x=seq_size, y=mean, color=method)) +
      geom_point(size=1.0) + geom_line(linewidth=0.5) +
      geom_hline(yintercept=0.7, linetype=2) + 
      geom_hline(yintercept=0.9, linetype=2) +
      scale_x_continuous(breaks=1:seq_size_test) +
      xlab('Sequence Size of Testing Set') + ylab('Accuracy') +
      guides(color=guide_legend(nrow=2, byrow=T)) +
      ggtitle(paste(sub_new_name, ', Training Set at Seq ', seq_train_iter, sep='')) + ylim(c(0, 1)) +
      theme(plot.title=element_text(hjust=0.5, size=10),
            panel.background=element_rect(fill = "white",
                                          colour = "black",
                                          linewidth = 0.5, linetype = "solid"),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            legend.position='bottom',
            legend.title=element_blank(),
            legend.text=element_text(size=10),
            legend.background=element_rect(fill='transparent', linewidth=0.2,
                                           color='white', linetype='solid'),
            plot.margin=margin(.2, .2, .2, .2, 'cm'))
    p_predict_test_iter_ls[[sub_new_name]] = p_predict_test_iter
  }
  
  
  # absolute measurement
  # remove K151 (no need to show in the appendix)
  p_predict_test_iter_arrange = marrangeGrob(
    p_predict_test_iter_ls[-5], ncol=4, nrow=2,
    top=paste('Testing Set at Seq ', seq_train_iter, sep='')
  )
  print(p_predict_test_iter_arrange)
  p_predict_iter_test_arrange_dir = file.path(
    parent_frt_predict_dir, prediction_name_2,
    paste('plot_xDAWN_cluster_2_predict_test_fix_seq_train_size_', seq_train_iter, '.png', sep='')
  )
  ggsave(p_predict_iter_test_arrange_dir, p_predict_test_iter_arrange, width=300, height=200, units='mm', dpi=300)
}









