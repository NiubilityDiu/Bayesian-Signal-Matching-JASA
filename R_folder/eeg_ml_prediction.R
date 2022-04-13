rm(list=ls(all.names=T))
# local_use = T
args <- commandArgs(trailingOnly = TRUE)
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

# include self-defined functions
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_path, 'Chapter_3', 'R_folder', 'self_R_fun', 'global_constant.R'))
frt_file_path = paste(parent_path, 'EEG_MATLAB_data', 'FRT_files', sep='/')

accuracy_df = data.frame(
  sub_name = NULL, method_name = NULL, letter = NULL, auc = NULL
)
method_names = c('FGMDM', 'RG_SVM', 'swLDA', 'XDAWN_LDA')
# sub_ids_2 = c("K106", "K107", "K108", "K111", "K154")
for (sub_name in sub_ids) {
  if (sub_name != 'M141') {
    print(sub_name)
    # M141 fails to produce SPD matrices.
    if (sub_name %in% names(FRT_file_name_ls)) {
      session_ids = FRT_file_name_ls[[sub_name]]
    } else {
      session_ids = c('001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT')
    }
    for(method_name in method_names) {
      print(method_name)
      total_correct = 0
      total_letter = 0
      total_auc = NULL
      for (session_id in session_ids) {
        result_dir = file.path(
          frt_file_path, sub_name, method_name,
          paste('*', session_id, '.json', sep='')
        )
        result_df = fromJSON(file=Sys.glob(result_dir))
        total_correct = total_correct + round(result_df$letter * result_df$num_letter, digits=0)
        total_letter = total_letter + result_df$num_letter
        total_auc = c(total_auc, result_df$auc)
      }
      total_accuracy = signif(total_correct / total_letter, digits=2) 
      accuracy_df = rbind.data.frame(
        accuracy_df, 
        data.frame(sub_name = sub_name, method_name = method_name, letter = total_accuracy,
                   auc = mean(total_auc))
      )
    }
  }
}

accuracy_df$method_name = factor(accuracy_df$method_name, levels=c('FGMDM', 'RG_SVM', 'XDAWN_LDA', 'swLDA'))

p_boxplot_method_letter = ggplot(data=accuracy_df, aes(x=method_name, y=letter)) +
  geom_boxplot() +
  xlab('') + ylab('Accuracy') +
  theme(plot.title=element_text(hjust=0.5),
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

p_boxplot_method_auc = ggplot(data=accuracy_df, aes(x=method_name, y=auc)) +
  geom_boxplot() +
  xlab('') + ylab('Accuracy') +
  theme(plot.title=element_text(hjust=0.5),
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

output_dir_1 = file.path(frt_file_path, 'ml_method_letter_boxplot.png')
ggsave(file.path(output_dir_1),
       p_boxplot_method_letter, width=300, height=200, units='mm', dpi=300)
output_dir_2 = file.path(frt_file_path, 'ml_method_auc_boxplot.png')
ggsave(file.path(output_dir_2),
       p_boxplot_method_auc, width=300, height=200, units='mm', dpi=300)

p_barplot_method_letter = ggplot(data=accuracy_df, aes(x=method_name, y=letter, fill=method_name)) +
  facet_wrap(~sub_name, nrow=5, ncol=11) + 
  geom_bar(stat="identity") + xlab('') + ylab('Accuracy') + 
  theme(plot.title=element_text(hjust=0.5),
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
p_barplot_method_auc = ggplot(data=accuracy_df, aes(x=method_name, y=auc, fill=method_name)) +
  facet_wrap(~sub_name, nrow=5, ncol=11) + 
  geom_bar(stat="identity") + xlab('') + ylab('Accuracy') + 
  theme(plot.title=element_text(hjust=0.5),
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

output_dir_3 = file.path(frt_file_path, 'ml_method_letter_barplot.png')
ggsave(file.path(output_dir_3),
       p_barplot_method_letter, width=700, height=300, units='mm', dpi=300)
output_dir_4 = file.path(frt_file_path, 'ml_method_auc_barplot.png')
ggsave(file.path(output_dir_4),
       p_barplot_method_auc, width=700, height=300, units='mm', dpi=300)


# p_scatter_single_cluster_ls = lapply(1:length(sub_ids), function(x) NULL)
# names(p_scatter_single_cluster_ls) = sub_ids
# ncd_summary_df = data.frame(
#   mean=NULL, sd=NULL, subject=NULL, cluster=NULL
# )
# for (sub_id in sub_ids) {
#   ncd_summary_sub_id_df = self_method_single_cluster_quantity(
#     ncd_1_exclude_summary, ncd_1_include_summary, sub_id
#   )
#   ncd_2_label_id = ncd_include_2_summary[[sub_id]]$Label
#   ncd_2_label_name_id = paste('Weight_clust_2_id_', ncd_2_label_id, '_001_BCI_TRN', sep="")
#   ncd_3_label_id = ncd_include_3_summary[[sub_id]]$Label
#   ncd_3_label_name_id = paste('Weight_clust_3_id_', ncd_3_label_id, '_001_BCI_TRN', sep="")
#   ncd_5_label_id = ncd_include_5_summary[[sub_id]]$Label
#   ncd_5_label_name_id = paste('Weight_clust_5_id_', ncd_5_label_id, '_001_BCI_TRN', sep="")
#   
#   ncd_summary_sub_id_df = rbind.data.frame(
#     ncd_summary_sub_id_df, data.frame(
#       mean=c(ncd_include_2_summary[[sub_id]][[ncd_2_label_name_id]]$Accuracy / 
#                ncd_include_2_summary[[sub_id]][[ncd_2_label_name_id]]$Total,
#              ncd_include_3_summary[[sub_id]][[ncd_3_label_name_id]]$Accuracy / 
#                ncd_include_3_summary[[sub_id]][[ncd_3_label_name_id]]$Total,
#              ncd_include_5_summary[[sub_id]][[ncd_5_label_name_id]]$Accuracy / 
#                ncd_include_5_summary[[sub_id]][[ncd_5_label_name_id]]$Total,
#              self_summary$self[self_summary$subject==sub_id]),
#       sd=c(0, 0, 0, 0),
#       subject=rep(sub_id,4),
#       cluster=c('ncd_2', 'ncd_3', 'ncd_5', 'self')
#     )
#   )
#   ncd_summary_df = rbind.data.frame(
#     ncd_summary_df, ncd_summary_sub_id_df
#   )
#   
#   ncd_summary_sub_id_df$cluster = factor(
#     ncd_summary_sub_id_df$cluster, 
#     levels=c('Exclude', 'Include', 'ncd_2', 'ncd_3', 'ncd_5', 'self'),
#     labels=c('1out', '1in', '2in', '3in', '5in', 'self')
#   )
#   value_diff = ncd_summary_sub_id_df$mean[ncd_summary_sub_id_df$cluster=='Self']-
#     ncd_summary_sub_id_df$mean[ncd_summary_sub_id_df$cluster=='Exclude']
#   value_diff = signif(value_diff, digits=2)
#   p_scatter_single_cluster = ggplot() +
#     geom_point(data=ncd_summary_sub_id_df,
#                aes(x=cluster, y=mean), shape=19, size=2, color='black') +
#     ylim(c(0, 1)) +
#     xlab('') + ylab('') + ggtitle(sub_id)
#   p_scatter_single_cluster_ls[[sub_id]] = p_scatter_single_cluster
# }
# p_scatter_single_cluster_combine = marrangeGrob(p_scatter_single_cluster_ls, nrow=5, ncol=11, top="")
# output_dir_5 = file.path(frt_file_path, 'ncd_size_1_accuracy_p3.png')
# ggsave(file.path(output_dir_5),
#        p_scatter_single_cluster_combine, width=600, height=300, units='mm', dpi=400)
# 
# 
# # boxplot
# ncd_summary_df$cluster = factor(
#   ncd_summary_df$cluster, 
#   levels=c('Exclude', 'Include', 'ncd_2', 'ncd_3', 'ncd_5', 'self'),
#   labels=c('1out', '1in', '2in', '3in', '5in', 'self')
# )
# p_boxplot_cluster = ggplot(data=ncd_summary_df, aes(x=cluster, y=mean)) +
#   geom_boxplot() + 
#   xlab('') + ylab('Accuracy')
# # p_boxplot_cluster
# output_dir_6 = file.path(frt_file_path, 'ncd_boxplot.png')
# ggsave(file.path(output_dir_6),
#        p_boxplot_cluster, width=200, height=100, units='mm', dpi=400)
