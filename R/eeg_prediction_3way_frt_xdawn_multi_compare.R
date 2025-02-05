# evaluate the prediction accuracy of 9 participants
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

seq_size_train = 5
seq_size_vec = 5:seq_size_train
target_char_train_size = 5
select_channel_ids = 1:16
select_channel_size = length(select_channel_ids)

parent_frt_predict_dir = file.path(parent_frt_dir, 'Prediction')

method_name = c('bkm_mixture_frt', 'mdwm_frt',
                'swLDA_reference_frt', 'smgp_frt', 
                'target_char_test_size')

target_char_id = length(method_name)
method_num = target_char_id - 1
xDAWN_result = lapply(1:(method_num+1), function(x) NULL)

prediction_name_2 = 'xDAWN_comp_2_reduced'
for (method_name_iter in method_name) {
  df_method_name_test_all_dir = file.path(parent_frt_predict_dir, prediction_name_2, 
                                          paste('K_', method_name_iter, '.RDS', sep=''))
  df_method_name_test_all = readRDS(df_method_name_test_all_dir)
  xDAWN_result[[method_name_iter]] = df_method_name_test_all
}

p_predict_test_ls = lapply(1:N, function(x) NULL)
names(p_predict_test_ls) = sub_name_vec

for (sub_new_name in sub_name_vec) {
  print(sub_new_name)
  
  seq_size_test = 0
  if (sub_new_name %in% names(FRT_file_name_ls)) {
    frt_common_name_vec = FRT_file_name_ls[[sub_new_name]]
  } else {
    frt_common_name_vec = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')
  }
  
  xDAWN_result_temp = xDAWN_result[['bkm_mixture_frt']]
  seq_size_test = lapply(
    1:length(frt_common_name_vec), function(x) nrow(xDAWN_result_temp[[sub_new_name]][[frt_common_name_vec[x]]]))
  seq_size_test = min(unlist(seq_size_test))
  
  p_predict_test_sub_ls = lapply(1:method_num, function(x) matrix(0, nrow=seq_size_test, ncol=seq_size_train))
  names(p_predict_test_sub_ls) = method_name[1:method_num]
  
  for (method_i in 1:method_num) {
    for (frt_common_name in frt_common_name_vec) {
      method_name_i = method_name[method_i]
      p_predict_test_sub_ls[[method_name_i]] = p_predict_test_sub_ls[[method_name_i]] +
        xDAWN_result[[method_name_i]][[sub_new_name]][[frt_common_name]][1:seq_size_test,] 
    }
    p_predict_test_sub_ls[[method_name_i]] = p_predict_test_sub_ls[[method_name_i]] / 
      sum(xDAWN_result[[method_name[target_char_id]]][[sub_new_name]])
  }

  p_predict_test_ls[[sub_new_name]] = p_predict_test_sub_ls
  
}



# produce the summary plot across participants
df_reduced_5_value = NULL
df_reduced_5_method = NULL
df_reduced_5_seq_size_test = NULL
df_reduced_5_subject_id = NULL

for (sub_new_name in sub_name_vec) {
  for (method_name_i in method_name[1:method_num]) {
    accuacy_5_iter_vec = p_predict_test_ls[[sub_new_name]][[method_name_i]][,5]
    seq_size_test_iter = length(accuacy_5_iter_vec)
    df_reduced_5_value = c(df_reduced_5_value, accuacy_5_iter_vec)
    df_reduced_5_method = c(df_reduced_5_method, rep(method_name_i, seq_size_test_iter))
    df_reduced_5_seq_size_test = c(df_reduced_5_seq_size_test, 1:seq_size_test_iter)
    df_reduced_5_subject_id = c(df_reduced_5_subject_id, rep(sub_new_name, seq_size_test_iter))
  }
}

reduced_train_seq_5_df = data.frame(
  value=df_reduced_5_value,
  method=df_reduced_5_method,
  seq_size_test=df_reduced_5_seq_size_test,
  subject_id=df_reduced_5_subject_id
)

# seq_test_max
reduced_train_seq_5_df = aggregate(
  x=cbind(seq_size_test, value)~method+subject_id, data=reduced_train_seq_5_df, FUN=max
)

reduced_train_seq_5_df$method = factor(
  reduced_train_seq_5_df$method, 
  levels=c('bkm_mixture_frt', 'mdwm_frt',
           'swLDA_reference_frt', 'smgp_frt'),
  labels=c('BSM-Mixture', 'MDWM', 'swLDA', 'SMGP')
)

# Boxplot
p_reduced_train_seq_5_boxplot = ggplot(data=reduced_train_seq_5_df, aes(x=method, y=value)) +
  geom_boxplot() + 
  xlab('\nMethod Name') + ylab('Character-level Accuracy\n') +
  stat_summary(fun='mean', aes(x=method, y=value), geom='point', shape=20, size=10, color='red') + 
  ggtitle('') + ylim(c(0, 1)) +
  theme(plot.title=element_text(hjust=0.5, size=18),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      linewidth = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='none',
        axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        plot.margin=margin(.3, .3, .3, .3, 'cm'))
print(p_reduced_train_seq_5_boxplot)

p_reduced_train_seq_5_boxplot_dir = file.path(
  parent_frt_predict_dir, prediction_name_2,
  paste('boxplot_predict_test_brain_damage_seq_train_size_5.png', sep='')
)
ggsave(p_reduced_train_seq_5_boxplot_dir, p_reduced_train_seq_5_boxplot,
       width=250, height=250, units='mm', dpi=300)




p_reduced_train_seq_5_boxplot_3 = ggplot(data=reduced_train_seq_5_df, aes(x=method, y=value)) +
  geom_point(aes(x=method, y=value, shape=factor(subject_id)), size=5.0) +
  geom_line(aes(x=method, y=value, group=factor(subject_id)), alpha=0.5, linetype='dashed') +
  scale_shape_manual(values=1:9) +
  geom_hline(yintercept=0.7, linetype=3) +
  stat_summary(fun='mean', aes(x=method, y=value), geom='point', shape=20, size=7.5, color='red') + 
  stat_summary(fun='mean', aes(x=method, y=value, group=1), geom='line', alpha=0.5, linetype='dashed', linewidth=1.5, color='red') +
  xlab('\nMethod Name') + ylab('Character-level Accuracy') +
  ggtitle('Participants with Brain Injuries or Neuro-degenerative Diseases') + ylim(c(0, 1)) +
  theme(plot.title=element_text(hjust=0.5, size=18),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      linewidth = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='none',
        axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        plot.margin=margin(.3, .3, .3, .3, 'cm'))
print(p_reduced_train_seq_5_boxplot_3)

p_reduced_train_seq_5_boxplot_3_dir = file.path(
  parent_frt_predict_dir, prediction_name_2,
  paste('spaghetti_plot_predict_test_brain_damage_seq_train_size_5.png', sep='')
)
ggsave(p_reduced_train_seq_5_boxplot_3_dir, 
       p_reduced_train_seq_5_boxplot_3, 
       width=240, height=120, units='mm', dpi=300)




