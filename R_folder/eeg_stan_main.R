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
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  rstan_options(auto_write=T)  # only for local running
  sub_new = 'K106'
  # sub_new = args[2]
  # seed_num  = 1
  letter_num_new = 6
} else {
  parent_dir = '/home/mtianwen'
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  rstan_options(auto_write=F)  # only for server running
  sub_new = K_sub_ids[as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))]
  # sub_new = args[2]
  # letter_num_new = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
  letter_num_new = as.integer(args[2])
}
total_sub_num = 20
seed_num = 612
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data')
data_type_trn = 'TRN_files'

sample_size_new = 12 * 15 * letter_num_new
weight_name_new = paste('Weight_001_BCI_TRN_New_Only_Letter_', letter_num_new, sep='')
new_data_output = compute_swlda_score(
  local_use, sub_new, data_type_trn, weight_name_new, sample_size_new
)
x_new = new_data_output$score
y_new = new_data_output$type

target_mean_new = mean(x_new[y_new==1])
ntarget_mean_new = mean(x_new[y_new==-1])
sd_new = sd(x_new)
# print(target_mean_new); print(ntarget_mean_new); print(sd_new)
# d_prime = (target_mean_new - ntarget_mean_new) / sd_new
# print(d_prime)



sub_source = setdiff(sub_ids[1:total_sub_num], sub_new)
size_length = length(sub_source)
size_source = rep(12 * 15 * 6, size_length)
weight_name_source = 'Weight_001_BCI_TRN'

# Collect source dataset (in a descending order with respect to the sample size)
x_source = y_source = index_source = NULL
for (id in 1:size_length) {
  print(id)
  source_id_output = compute_swlda_score(
    local_use, sub_source[id], data_type_trn, weight_name_source, size_source[id]
  )
  score_id = source_id_output$score
  type_id = source_id_output$type
  x_source = c(x_source, score_id)
  y_source = c(y_source, type_id)
  index_source = c(index_source, rep(id, size_source[id]))
  # print(sd(score_id))
}

density_data_df = data.frame(
  x = c(x_new, x_source),
  y = c(y_new, y_source),
  index = c(rep(total_sub_num, length(x_new)), index_source)
)
mean(density_data_df$x[density_data_df$y == 1]) - mean(density_data_df$x[density_data_df$y == -1])
eta_mean = 1 / sapply(1:total_sub_num, function(id) var(density_data_df$x[density_data_df$index == id]))

stan_data_input = list(
  size_source = size_source,
  x_source = x_source,
  y_source = y_source,
  index_source = index_source,
  x_new = x_new,
  y_new = y_new,
  sub_n_source = length(size_source),
  total_size_source = length(y_source),
  size_new = length(y_new),
  mu_source_diff_alpha = rep(7, total_sub_num-1),
  mu_source_diff_beta = rep(10, total_sub_num-1),
  mu_new_diff_alpha = 7, mu_new_diff_beta = 10,
  eta_source_alpha = floor(2 * eta_mean[1:(total_sub_num-1)]),
  eta_source_beta = rep(2, total_sub_num-1),
  eta_new_alpha = floor(2 * eta_mean[total_sub_num]),
  eta_new_beta = 2
)

stan_model_obj = stan_model(
  file=Sys.glob(file.path(parent_dir_r, 'stan', 'participant_selection.stan')), model_name='select'
)

stan_model_fit = sampling(
  stan_model_obj, data=stan_data_input, chains=2, seed=seed_num, iter=300, warmup=200
)

dir.create(file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output'))
file_suffix = paste('letter_num_', letter_num_new, sep='')
stan_output_dir_1 = file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output', 
                              paste('mcmc_summary_', file_suffix, '.txt', sep=''))
sink(file=stan_output_dir_1)
print(stan_model_fit)
sink()

stan_output_dir_2 = file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output', 
                              paste('mcmc_summary_', file_suffix, '.rds', sep=''))
saveRDS(extract(stan_model_fit), stan_output_dir_2)

# stan_model_fit = readRDS(stan_output_dir_2)
# phi_mcmc = stan_model_fit$phi_source
# phi_mcmc = extract(stan_model_fit, 'phi_source')$phi_source
# phi_mean = signif(apply(phi_mcmc, 2, mean), digits=2)
# 
# p_density = ggplot(data=density_data_df) + 
#   geom_density(aes(x=x, color=factor(y, levels=c(1, -1), labels=c('Target', 'Non-target')))) +
#   facet_wrap(~factor(index, levels=1:total_sub_num, 
#                      labels=c(paste('Source: ', sub_source, ', Phi: ', phi_mean, sep=''), 
#                               paste('New: ', sub_new, sep=''))), 
#              nrow=5, ncol=4) +
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
# density_plot_dir = file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output', 
#                              paste('density_plot_', file_suffix, '.png', sep=''))
# ggsave(density_plot_dir, p_density, width=300, height=300, units='mm', dpi=300)













# #####
# sub_source = K_sub_ids[1:20]
# size_length = length(sub_source)
# size_source = rep(12*15*5, size_length)
# # Collect source dataset
# x_source = y_source = index_source = NULL
# target_mean_source = ntarget_mean_source = sd_source = NULL
# for (id in 1:size_length) {
#   print(id)
#   source_id_output = compute_swlda_score(
#     local_use, sub_source[id], data_type_trn, 'Weight_001_BCI_TRN.mat', size_source[id]
#   )
#   score_id = source_id_output$score
#   type_id = source_id_output$type
#   x_source = c(x_source, score_id)
#   y_source = c(y_source, type_id)
#   index_source = c(index_source, rep(id, size_source[id]))
#   target_mean_source = c(target_mean_source, mean(score_id[type_id==1]))
#   ntarget_mean_source = c(ntarget_mean_source, mean(score_id[type_id==-1]))
#   sd_source = c(sd_source, sd(score_id))
# }
# 
# 
# par(mfrow=c(2,1))
# plot(ntarget_mean_source, target_mean_source, type='p', xlim=c(-0.5, 0.5), ylim=c(0, 1.5),
#      xlab='Non-target Mean', ylab='Target Mean')
# plot(sd_source, target_mean_source, type='p', xlim=c(0, 1), ylim=c(0, 1.5),
#      xlab='Standard Deviation', ylab='Target Mean')
# 
# K_sub_10_df = data.frame(
#   score = x_source,
#   type = y_source,
#   sub_id = index_source
# )
# 
# p_density = ggplot(K_sub_10_df, 
#                    aes(x=x_source, color=factor(y_source, levels=c(1,-1), 
#                                                 labels=c('Target', 'Non-target')))) +
#   geom_density(size=1.5) + facet_wrap(~factor(sub_id, levels=1:10, labels=K_sub_ids[1:10]), nrow=5, ncol=2) +
#   xlab('Classifier Score')  + ylab('') + ggtitle('Density Plot') +
#   theme(plot.title=element_text(hjust=0.5),
#         panel.background=element_rect(fill = "white",
#                                       colour = "black",
#                                       size = 0.5, linetype = "solid"),
#         panel.grid.major=element_blank(),
#         panel.grid.minor=element_blank(),
#         legend.position='bottom',
#         legend.title=element_blank(),
#         legend.background=element_rect(fill='transparent', size=0.2,
#                                        color='white', linetype='solid'),
#         plot.margin=margin(.25, .25, .25, .25, 'cm'))
# p_density
