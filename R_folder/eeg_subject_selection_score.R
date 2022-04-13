# rm(list=ls(all.names=T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T' || args[1] == 'True')

library(R.matlab)
library(ggplot2)
library(gridExtra)
library(rjson)

if (local_use) {
  parent_path = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
} else {
  parent_path = '/home/mtianwen'
}
seed_num = 612
set.seed(seed_num)
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'self_R_fun', 'global_constant.R'))
total_sub_num = 20
total_letter_num = 10

frt_file_path = file.path(parent_path, 'EEG_MATLAB_data', 'FRT_files')
sub_pool = sub_ids[1:total_sub_num]


methods = c('Merged_Letter', 'Only_Letter')
binary_accuracy = se_accuracy = sp_accuracy = ppv_accuracy = letter_accuracy = 
  array(0, dim=c(total_letter_num, 2, total_sub_num))

for (sub_id in 1:total_sub_num) {
  for (m_id in 1:2) {
    for (l_id in 2:total_letter_num) {
      json_file_dir = file.path(frt_file_path, sub_pool[sub_id], 'swLDA', 
                                paste('Result_New_', methods[m_id], '_', l_id, '.json', sep=''))
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
  method = rep(rep(c('Merge', 'Itself'), each=total_letter_num), total_sub_num),
  letter_size = rep(1:total_letter_num, total_sub_num * 2)
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

p_se_dir = file.path(frt_file_path, 'Sub_Total_20', 'p_se.png')
ggsave(p_se_dir, p_se, width=200, height=250, units='mm', dpi=300)

p_sp_dir = file.path(frt_file_path, 'Sub_Total_20', 'p_sp.png')
ggsave(p_sp_dir, p_sp, width=200, height=250, units='mm', dpi=300)

p_ppv_dir = file.path(frt_file_path, 'Sub_Total_20', 'p_ppv.png')
ggsave(p_ppv_dir, p_ppv, width=200, height=250, units='mm', dpi=300)

p_binary_dir = file.path(frt_file_path, 'Sub_Total_20', 'p_binary.png')
ggsave(p_binary_dir, p_binary, width=200, height=250, units='mm', dpi=300)

p_letter_dir = file.path(frt_file_path, 'Sub_Total_20', 'p_letter.png')
ggsave(p_letter_dir, p_letter, width=200, height=250, units='mm', dpi=300)
