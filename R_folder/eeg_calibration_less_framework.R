# rm(list=ls(all.names=T))
# local_use = T
args <- commandArgs(trailingOnly = TRUE)
local_use = (args[1] == 'T' || args[1] == 'True')

library(R.matlab)
library(ggplot2)
library(gridExtra)
library(rjson)

if (local_use) {
  parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  letter_num_source = 5
  sub_new_id = 4
} else {
  parent_dir = '/home/mtianwen'
  letter_num_source = as.integer(args[2])
  sub_new_id = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
}
parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'TRN_files')
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))

seed_num = 612
set.seed(seed_num)
data_type = 'TRN_files'
sub_pool_size = 20
sub_pool = sub_ids[1:sub_pool_size]
sub_new = sub_pool[sub_new_id]
sub_source_pool = setdiff(sub_pool, sub_new)
sub_source_size = sub_pool_size - 1
threshold_value = 0.02

# Fit swLDA to source subject (New Only)
# The part of code is realized in MATLAB.


# Select subject based on certain criterion (with thresholds)
# 
# /home/mtianwen/Chapter_3/script/eeg_R_subject_selection.pbs
if (1) {
  for (letter_num_new in 2:10) {
    sample_size_new = 12 * 15 * letter_num_new
    sample_size_source = 12 * 15 * letter_num_source
    weight_name = paste('Weight_001_BCI_TRN_New_Only_Letter_', letter_num_new, sep='')
    
    # new_data + new_weight = new_score
    dist_target = rep(0, sub_pool_size)
    dist_ntarget = rep(0, sub_pool_size)
    data_new_obj = import_data_obj(local_use, sub_new, data_type)
    swlda_new_obj = import_swlda_obj(local_use, sub_new, data_type, weight_name)
    swlda_new_output_obj = compute_swlda_score(data_new_obj, swlda_new_obj, sample_size_new)
    swlda_new_summary_obj = compute_swlda_score_summary(swlda_new_output_obj)
    
    for (i in 1:sub_pool_size) {
      sub_id = sub_pool[i]
      # print(sub_id)
      if (sub_id != sub_new) {
        data_source_obj = import_data_obj(local_use, sub_id, data_type)
        swlda_source_obj = import_swlda_obj(local_use, sub_id, data_type, weight_name)
        swlda_source_output_obj = compute_swlda_score(data_source_obj, swlda_source_obj, sample_size_source)
        swlda_source_summary_obj = compute_swlda_score_summary(swlda_source_output_obj)
        dist_target[i] = compute_kl_divergence_normal(
          swlda_source_summary_obj$target_mean, swlda_source_summary_obj$std,
          swlda_new_summary_obj$target_mean, swlda_new_summary_obj$std
        )
        dist_ntarget[i] = compute_kl_divergence_normal(
          swlda_source_summary_obj$ntarget_mean, swlda_source_summary_obj$std,
          swlda_new_summary_obj$ntarget_mean, swlda_new_summary_obj$std
        )
      }
    }
    
    print(dist_target)
    print(dist_ntarget)
    
    select_subject = sub_pool[dist_target <= threshold_value & dist_ntarget <= threshold_value]
    select_subject = setdiff(select_subject, sub_new)
    print(select_subject)
    
    dir.create(
      file.path(parent_eeg_data_dir, sub_new, 'selection_output')
    )
    dir.create(
      file.path(parent_eeg_data_dir, sub_new, 'selection_output', 
                paste('source_letter_num_', letter_num_source, sep=''))
    )
    select_subject_dir = file.path(
      parent_eeg_data_dir, sub_new, 'selection_output', 
      paste('source_letter_num_', letter_num_source, sep=''),
      paste('select_new_letter_num_', letter_num_new, '.mat', sep=''))
    writeMat(
      select_subject_dir, select_subject = select_subject, new_subject = sub_new,
      threshold = threshold_value
    )
  }
}


# Refit the swLDA with selected subjects' data
# The part of code is realized in MATLAB.
# EEGCalibrationLessRefit.m
# /home/mtianwen/Chapter_3/script/eeg_MATLAB_swlda_cluster_fit.pbs


# Predict on FRT files
# Record SE, SP, PPV, and character-level prediction accuracy.
# The part of code is realized in Python.
# EEG_swLDA_predict.py
# /home/mtianwen/Chapter_3/script/eeg_Python_swlda_cluster_predict.pbs


# Visualize the prediction accuracy
if (0) {
  
  frt_file_path = file.path(parent_dir, 'EEG_MATLAB_data', 'FRT_files')
  methods = c('Merged_Letter', 'Only_Letter')
  total_letter_num = 10
  binary_accuracy = se_accuracy = sp_accuracy = ppv_accuracy = letter_accuracy = 
    array(0, dim=c(total_letter_num, 2, sub_pool_size))
  
  for (sub_id in 1:sub_pool_size) {
    sub_name_id = sub_pool[sub_id]
    for (m_id in 1:2) {
      method_id = methods[m_id]
      for (l_id in 2:total_letter_num) {
        if (method_id == 'Only_Letter') {
          json_file_dir = file.path(frt_file_path, sub_name_id, 'swLDA', 
                                    paste('Result_New_', methods[m_id], '_', l_id, '.json', sep=''))
        } else {
          json_file_dir = file.path(
            frt_file_path, sub_name_id, 'swLDA', paste('source_letter_num_', letter_num_source, sep=''),
            paste('Result_New_', methods[m_id], '_', l_id, '_KL_Threshold_', threshold_value, '.json', sep='')
          )
        }
        
        accuracy_summary = fromJSON(file=Sys.glob(json_file_dir))
        print(accuracy_summary)
        frt_name_vec = names(accuracy_summary)
        total_char = 0
        total_tp = total_tn = total_se_denom = total_sp_denom = total_ppv_denom = 0
        for (frt_name in frt_name_vec) {
          binary_accuracy[l_id, m_id, sub_id] = binary_accuracy[l_id, m_id, sub_id] +
            accuracy_summary[[frt_name]]$binary * accuracy_summary[[frt_name]]$num_letter
          letter_accuracy[l_id, m_id, sub_id] = letter_accuracy[l_id, m_id, sub_id] +
            accuracy_summary[[frt_name]]$letter * accuracy_summary[[frt_name]]$num_letter
          total_char = total_char + accuracy_summary[[frt_name]]$num_letter
          
          total_tp = total_tp + accuracy_summary[[frt_name]]$tp
          total_tn = total_tn + accuracy_summary[[frt_name]]$tn
          total_se_denom = total_se_denom + accuracy_summary[[frt_name]]$tp + accuracy_summary[[frt_name]]$fn
          total_sp_denom = total_sp_denom + accuracy_summary[[frt_name]]$tn + accuracy_summary[[frt_name]]$fp
          total_ppv_denom = total_ppv_denom + accuracy_summary[[frt_name]]$tp + accuracy_summary[[frt_name]]$fp
        }
        binary_accuracy[l_id, m_id, sub_id] = binary_accuracy[l_id, m_id, sub_id] / total_char
        letter_accuracy[l_id, m_id, sub_id] = letter_accuracy[l_id, m_id, sub_id] / total_char
        
        se_accuracy[l_id, m_id, sub_id] = total_tp / total_se_denom
        sp_accuracy[l_id, m_id, sub_id] = total_tn / total_sp_denom
        ppv_accuracy[l_id, m_id, sub_id] = total_tp / total_ppv_denom
      }
    }
  }
  
  accuracy_long_df = data.frame(
    se = as.numeric(se_accuracy),
    sp = as.numeric(sp_accuracy),
    ppv = as.numeric(ppv_accuracy),
    binary = as.numeric(binary_accuracy),
    letter = as.numeric(letter_accuracy),
    subject = rep(sub_pool, each=2*total_letter_num),
    method = rep(rep(c('Merge', 'Itself'), each=total_letter_num), sub_pool_size),
    letter_size = rep(1:total_letter_num, sub_pool_size * 2)
  )
  
  p_binary = ggplot(data=accuracy_long_df, aes(x=letter_size, y=binary, color=method)) +
    facet_wrap(~subject, nrow=5, ncol=4) +
    geom_point(shape=19, size=2) + geom_line() +
    ylim(c(0, 1)) +
    scale_x_continuous(breaks=1:total_letter_num, labels=1:total_letter_num) +
    xlab('Sample Size (# of Letters for New Subject)') + ylab('') + ggtitle('Binary Accuracy') +
    theme(plot.title=element_text(hjust=0.5, size=15),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.25, .25, .25, .25, 'cm'))
  p_binary
  
  p_se = ggplot(data=accuracy_long_df, aes(x=letter_size, y=se, color=method)) +
    facet_wrap(~subject, nrow=5, ncol=4) +
    geom_point(shape=19, size=2) + geom_line() +
    ylim(c(0, 1)) +
    scale_x_continuous(breaks=1:total_letter_num, labels=1:total_letter_num) +
    xlab('Sample Size (# of Letters for New Subject)') + ylab('') + ggtitle('Sensitivity (Recall)') +
    theme(plot.title=element_text(hjust=0.5, size=15),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.25, .25, .25, .25, 'cm'))
  p_se
  
  p_sp = ggplot(data=accuracy_long_df, aes(x=letter_size, y=sp, color=method)) +
    facet_wrap(~subject, nrow=5, ncol=4) +
    geom_point(shape=19, size=2) + geom_line() +
    ylim(c(0, 1)) +
    scale_x_continuous(breaks=1:total_letter_num, labels=1:total_letter_num) +
    xlab('Sample Size (# of Letters for New Subject)') + ylab('') + ggtitle('Specificity') +
    theme(plot.title=element_text(hjust=0.5, size=15),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.25, .25, .25, .25, 'cm'))
  p_sp
  
  p_ppv = ggplot(data=accuracy_long_df, aes(x=letter_size, y=ppv, color=method)) +
    facet_wrap(~subject, nrow=5, ncol=4) +
    geom_point(shape=19, size=2) + geom_line() +
    ylim(c(0, 1)) +
    scale_x_continuous(breaks=1:total_letter_num, labels=1:total_letter_num) +
    xlab('Sample Size (# of Letters for New Subject)') + ylab('') + ggtitle('Positive Predicitve Value (Precision)') +
    theme(plot.title=element_text(hjust=0.5, size=15),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.25, .25, .25, .25, 'cm'))
  p_ppv
  
  p_letter = ggplot(data=accuracy_long_df, aes(x=letter_size, y=letter, color=method)) +
    facet_wrap(~subject, nrow=5, ncol=4) +
    geom_point(shape=19, size=2) + geom_line() +
    ylim(c(0, 1)) +
    scale_x_continuous(breaks=1:total_letter_num, labels=1:total_letter_num) +
    xlab('Sample Size (# of Letters for New Subject)') + ylab('') + ggtitle('Character Accuracy') +
    theme(plot.title=element_text(hjust=0.5, size=15),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.2,
                                         color='white', linetype='solid'),
          plot.margin=margin(.25, .25, .25, .25, 'cm'))
  p_letter
  
  dir.create(file.path(frt_file_path, 'Sub_Total_20'))
  source_letter_num_dir = file.path(
    frt_file_path, 'Sub_Total_20', paste('source_letter_num_', letter_num_source, sep='')
  )
  dir.create(source_letter_num_dir)
  common_suffix_name = paste(
    'New_Merged_Letter_2_10_KL_Threshold_', 
    threshold_value, '.png', sep=''
  )
  
  p_se_dir = file.path(source_letter_num_dir, paste('p_1_se_', common_suffix_name, sep=''))
  ggsave(p_se_dir, p_se, width=200, height=250, units='mm', dpi=300)
  
  p_sp_dir = file.path(source_letter_num_dir, paste('p_2_sp_', common_suffix_name, sep=''))
  ggsave(p_sp_dir, p_sp, width=200, height=250, units='mm', dpi=300)
  
  p_ppv_dir = file.path(source_letter_num_dir, paste('p_3_ppv_', common_suffix_name, sep=''))
  ggsave(p_ppv_dir, p_ppv, width=200, height=250, units='mm', dpi=300)
  
  p_binary_dir = file.path(source_letter_num_dir, paste('p_4_binary_', common_suffix_name, sep=''))
  ggsave(p_binary_dir, p_binary, width=200, height=250, units='mm', dpi=300)
  
  p_letter_dir = file.path(source_letter_num_dir, paste('p_5_letter_', common_suffix_name, sep=''))
  ggsave(p_letter_dir, p_letter, width=200, height=250, units='mm', dpi=300)
}