rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
local_use = T
# local_use = (args[1] == 'T' || args[1] == 'True')
library(rstan)
library(R.matlab)
library(ggplot2)

if (local_use) {
  parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  # parent_dir = 'K:\\Dissertation\\Dataset and Rcode'
  # parent_dir = 'C:/Users/mtwen/Downloads'
  rstan_options(auto_write=T)  # only for local running
  letter_num_new = 2
} else {
  parent_dir = '/home/mtianwen'
  rstan_options(auto_write=F)  # only for server running
  letter_num_new = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
}
parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))

total_sub_num = 20
seed_num = 612
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data', 'TRN_files')

file_suffix = paste('letter_num_', letter_num_new, sep='')
selection_matrix = matrix(1, nrow=total_sub_num, ncol=total_sub_num)
sub_pool = sub_ids[1:total_sub_num]

for (i in 1:total_sub_num) {
  stan_output_dir = file.path(parent_eeg_data_dir, sub_pool[i], 'stan_output', 
                              paste('mcmc_summary_', file_suffix, '.rds', sep=''))
  stan_model_fit = readRDS(stan_output_dir)
  if (is.null(stan_model_fit)) {
    selection_matrix[i, ] = 0
    selection_matrix[i, i] = 1
  } else {
    phi_mcmc = stan_model_fit$phi_source
    phi_mean = apply(phi_mcmc, 2, mean)
    if (i == 1) {
      selection_matrix[1,2:total_sub_num] = phi_mean
    } else if (i == total_sub_num) {
      selection_matrix[total_sub_num,1:(total_sub_num-1)] = phi_mean
    } else {
      selection_matrix[i,1:(i-1)] = phi_mean[1:(i-1)]
      selection_matrix[i,(i+1):total_sub_num] = phi_mean[i:(total_sub_num-1)]
    } 
  }
}

# Dummy data
x = sub_pool
y = rev(sub_pool)
dat_heatmap = expand.grid(X=x, Y=y)
dat_heatmap$Z = as.numeric(t(selection_matrix))
# dat_heatmap$Z_discrete = as.numeric(t(selection_matrix>0.6))

p_heatmap = ggplot(dat_heatmap, aes(X, rev(Y), fill=Z)) + geom_tile(color = "gray") + 
  geom_text(label = signif(dat_heatmap$Z, digits=2))  + 
  scale_fill_gradient(low="red", high="blue") + xlab('Source Subject') + ylab('New Subject') +
  theme(plot.title=element_text(hjust=0.5, size=6),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='none',
        legend.title=element_blank(),
        legend.background=element_rect(fill='transparent', size=0.2,
                                       color='white', linetype='solid'),
        plot.margin=margin(.25, .25, .25, .25, 'cm'))
p_heatmap

# p_heatmap_discrete = ggplot(dat_heatmap, aes(X, rev(Y), fill=Z_discrete)) + geom_tile(color = "gray") + 
#   geom_text(label = signif(dat_heatmap$Z_discrete, digits=2))  + 
#   scale_fill_gradient(low="red", high="blue") + xlab('Source Subject') + ylab('New Subject') +
#   theme(plot.title=element_text(hjust=0.5, size=10),
#         panel.background=element_blank(),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         legend.position='none',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.25, .25, .25, .25, 'cm'))
# p_heatmap_discrete

dir.create(file.path(parent_eeg_data_dir, paste('Sub_Total', total_sub_num, sep='_')))
heatmap_c_dir = file.path(parent_eeg_data_dir, paste('Sub_Total', total_sub_num, sep='_'),
                        paste('heatmap_continuous_', file_suffix, '.png', sep=''))
# heatmap_d_dir = file.path(parent_eeg_data_dir, paste('Sub_Total', total_sub_num, sep='_'),
#                           paste('heatmap_discrete_0.5_', file_suffix, '.png', sep=''))
ggsave(heatmap_c_dir, p_heatmap, width=600, height=200, units='mm', dpi=300)
# ggsave(heatmap_d_dir, p_heatmap_discrete, width=200, height=150, units='mm', dpi=300)

matrix_dir = file.path(parent_eeg_data_dir, paste('Sub_Total', total_sub_num, sep='_'),
                       paste('selection_matrix_', file_suffix, '.mat', sep=''))
writeMat(matrix_dir, matrix=selection_matrix)
