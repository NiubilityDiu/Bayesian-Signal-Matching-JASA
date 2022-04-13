rm(list=ls(all.names=T))
local_use = T
args <- commandArgs(trailingOnly = TRUE)
# local_use = (args[1] == 'T' || args[1] == 'True')

library(R.matlab)
library(ggplot2)
library(gridExtra)


if (local_use) {
  parent_path = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  # method_name = 'gmm_cluster_3_label'
  method_name = paste('ncd_cluster_', as.integer(args[2]), '_label', sep="")
} else {
  parent_path = '/home/mtianwen'
  method_name = args[2]
}

# for plot arrangment
nrow = as.integer(args[3])
ncol = as.integer(args[4])
# nrow = 3; ncol = 1

# include self-defined functions
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'self_defined_fun.R'))
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'global_constant.R'))
frt_file_path = paste(parent_path, 'EEG_MATLAB_data', 'FRT_files', sep='/')

cluster_assign_dir = file.path(
  parent_path, 'EEG_MATLAB_data', 'TRN_files', 'Dist_matrix', 
  paste(method_name, '.mat', sep = '')
)
cluster_assign = readMat(Sys.glob(cluster_assign_dir))
cluster_assign$label = t(cluster_assign$label)
cluster_unique = sort(unique(cluster_assign$label))
cluster_size = length(cluster_unique)

# import the rds file
# n_to_select = 3
accuracy_dir = file.path(
  parent_path, 'EEG_MATLAB_data', 'TRN_files', 'Dist_matrix',
  paste(method_name, '_accuracy_comparison.rds', sep="")
)
accuracy_compare = readRDS(accuracy_dir)


subject_id = NULL
session_id = NULL
num_letter = NULL
seq_num = NULL
value = NULL
method = NULL


for (i in 1:length(sub_ids)) {
  sub_name = sub_ids[i]
  # print(sub_name)
  if (sub_name %in% names(FRT_file_name_ls)) {
    frt_file_name_vec = FRT_file_name_ls[[sub_name]]
  } else {
    frt_file_name_vec = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')
  }
  sub_name_ls = accuracy_compare[[sub_name]]
  for (frt_file_name in frt_file_name_vec) {
    sub_name_frt_file_ls = sub_name_ls[[frt_file_name]]
    value = c(value, sub_name_frt_file_ls[['self']])
    frt_file_name_unique = names(sub_name_frt_file_ls)
    cluster_ids_unique = grep('cluster', frt_file_name_unique)
    for (cluster_id in cluster_ids_unique) {
      value = c(value, sub_name_frt_file_ls[[frt_file_name_unique[cluster_id]]])
    }
    file_rep_time = length(frt_file_name_unique)-2
    subject_id = c(subject_id, rep(sub_name, sub_name_frt_file_ls$num_rep * file_rep_time))
    session_id = c(session_id, rep(frt_file_name, sub_name_frt_file_ls$num_rep * file_rep_time))
    num_letter = c(num_letter, rep(sub_name_frt_file_ls$num_letter, sub_name_frt_file_ls$num_rep * file_rep_time))
    seq_num = c(seq_num, rep(1:sub_name_frt_file_ls$num_rep, file_rep_time))
    method = c(method, rep(c('self', frt_file_name_unique[cluster_ids_unique]), each=sub_name_frt_file_ls$num_rep))
  }
}


accuracy_df = data.frame(
  subject_id = subject_id,
  session_id = session_id,
  num_letter = num_letter,
  seq_num = seq_num,
  value = value,
  method = method
)

accuracy_df_2 = aggregate(
  accuracy_df, 
  list(accuracy_df$subject_id,
       accuracy_df$session_id,
       accuracy_df$method), tail, n=1
)[, -(1:3)]

accuracy_df_3 = aggregate(
  cbind(num_letter, value)
  ~ subject_id + method, accuracy_df_2, sum
)

label_rep = NULL
for (sub_name in accuracy_df_3$subject_id) {
  label_rep = c(label_rep, cluster_assign$label[cluster_assign$name==sub_name])
  # sub_name_rep = c(sub_name_rep, rep(sub_name, sum(accuracy_df_3$subject_id == sub_name)))
}
accuracy_df_3$label = label_rep
accuracy_df_3$percent = accuracy_df_3$value / accuracy_df_3$num_letter

accuracy_df_3_baseline = accuracy_df_3[accuracy_df_3$method=='self', c('subject_id', 'method', 'percent')]
accuracy_df_4 = merge(accuracy_df_3, accuracy_df_3_baseline, by="subject_id")
accuracy_df_4$percent_relative = accuracy_df_4$percent.x - accuracy_df_4$percent.y
accuracy_df_4 = accuracy_df_4[, c('subject_id', 'method.x', 'num_letter', 'value', 'label', 'percent.x', 'percent_relative')]
names(accuracy_df_4) = c('subject_id', 'method', 'num_letter', 'value', 'label', 'percent', 'percent_relative')
accuracy_df_4 = accuracy_df_4[accuracy_df_4$method != 'self',]

# We only look at the value with the largest sequence number
p_scatter_ls = lapply(1:cluster_size, function(x) NULL)
names(p_scatter_ls) = paste('label ', 0:(cluster_size-1), sep="")
color_fill_init = c(rep("#999999", cluster_size-1), "#56B4E9")
for (predict_label in 0:(cluster_size-1)) {
  accuracy_df_3_label = accuracy_df_3[accuracy_df_3$label == predict_label,]
  accuracy_df_3_label$method = factor(accuracy_df_3_label$method, 
                                      labels=c(paste('id', 0:(cluster_size-1), sep='_'), 'self'))
  color_fill_label = append(color_fill_init, "#E69F00", predict_label)
  p_scatter_id = ggplot(data=accuracy_df_3_label, aes(x=subject_id, y=percent, fill=method)) +
    geom_bar(stat='identity', position=position_dodge()) + 
    scale_fill_manual(values=color_fill_label) + 
    # facet_wrap(~factor(label), nrow=3) + 
    # geom_point(data=accuracy_df_3_label[accuracy_df_3_label$method=='self',], 
    #            aes(x=subject_id, y=percent), shape=1, size=3, color='red') +
    # geom_point(data=accuracy_df_3_label[accuracy_df_3_label$method==paste('cluster_', cluster_size, '_id_', predict_label, sep=""),], 
    #            aes(x=subject_id, y=percent), shape=1, size=3, color='blue') +
    ylim(c(0, 1)) +
    xlab('') + ylab('') + ggtitle(paste('Label ', predict_label, sep="")) +
    theme(# plot.title = element_text(hjust = 0.5), 
          legend.title=element_blank(), 
          legend.text=element_text(size=8), legend.key.size = unit(5,'mm')) 
    # theme_minimal()
  p_scatter_ls[[paste('label ', predict_label, sep="")]] = p_scatter_id
} 
p_scatter_combine = marrangeGrob(p_scatter_ls, nrow=nrow, ncol=ncol, top='Accuracy')
# p_scatter_combine
output_dir = file.path(parent_path, 'EEG_MATLAB_data', 'FRT_files',
                   paste('swLDA_accuracy_', method_name, '_plot.png', sep=""))
ggsave(file.path(output_dir),
       p_scatter_combine, width=450, height=200, units='mm', dpi=400)


# We only look at the value with the largest sequence number
p_scatter_diff_ls = lapply(1:cluster_size, function(x) NULL)
names(p_scatter_diff_ls) = paste('label ', 0:(cluster_size-1), sep="")
color_fill_init = rep("#56B4E9", cluster_size-1)
for (predict_label in 0:(cluster_size-1)) {
  accuracy_df_4_label = accuracy_df_4[accuracy_df_4$label == predict_label,]
  accuracy_df_4_label$method = factor(accuracy_df_4_label$method, 
                                      labels=paste('id', 0:(cluster_size-1), sep='_'))
  color_fill_label = append(color_fill_init, "#E69F00", predict_label)
  p_scatter_diff_id = ggplot(data=accuracy_df_4_label, aes(x=subject_id, y=percent_relative, fill=method)) +
    geom_bar(stat='identity', position=position_dodge()) + 
    scale_fill_manual(values=color_fill_label) + 
    # facet_wrap(~factor(label), nrow=3) + 
    # geom_point(data=accuracy_df_3_label[accuracy_df_3_label$method=='self',], 
    #            aes(x=subject_id, y=percent), shape=1, size=3, color='red') +
    # geom_point(data=accuracy_df_3_label[accuracy_df_3_label$method==paste('cluster_', cluster_size, '_id_', predict_label, sep=""),], 
    #            aes(x=subject_id, y=percent), shape=1, size=3, color='blue') +
    ylim(c(-1, 0.5)) +
    xlab('') + ylab('') + ggtitle(paste('Label ', predict_label, sep="")) +
    theme(# plot.title = element_text(hjust = 0.5), 
      legend.title=element_blank(), 
      legend.text=element_text(size=8), legend.key.size = unit(5,'mm')) 
  # theme_minimal()
  p_scatter_diff_ls[[paste('label ', predict_label, sep="")]] = p_scatter_diff_id
} 
p_scatter_diff_combine = marrangeGrob(p_scatter_diff_ls, nrow=nrow, ncol=ncol, top='Relative Accuracy')
# p_scatter_diff_combine
output_diff_dir = file.path(parent_path, 'EEG_MATLAB_data', 'FRT_files',
                       paste('swLDA_accuracy_', method_name, '_relative_plot.png', sep=""))
ggsave(file.path(output_diff_dir),
       p_scatter_diff_combine, width=450, height=200, units='mm', dpi=400)

print('Cluster criterion:')
print(self_cluster_quantitiy(accuracy_df_4))