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
  sub_new = 'K107'
  letter_num_new = 4
} else {
  parent_dir = '/home/mtianwen'
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  rstan_options(auto_write=F)  # only for server running
  sub_new = K_sub_ids[as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))]
  letter_num_new = as.integer(args[2])
}
source(file.path(parent_dir_r, 'self_R_fun', 'gibbs_sampling_fun.R'))

total_sub_num = 20
seed_num = 612
set.seed(seed_num)
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data')
data_type_trn = 'TRN_files'

sample_size_new = 12 * 15 * letter_num_new
weight_name_new = paste('Weight_001_BCI_TRN_New_Only_Letter_', letter_num_new, sep='')
new_data_output = compute_swlda_score(
  local_use, sub_new, data_type_trn, weight_name_new, sample_size_new
)
new_data_ls = list(
  score=new_data_output$score, type=new_data_output$type, 
  size_tar=sum(new_data_output$type==1), size_ntar=sum(new_data_output$type != 1)
)

sub_source = setdiff(sub_ids[1:total_sub_num], sub_new)
size_length = length(sub_source)
size_source = rep(12 * 15 * 5, size_length)
weight_name_source = 'Weight_001_BCI_TRN'

# Collect source dataset (in a descending order with respect to the sample size)
source_data_ls = lapply(1:size_length, function(x) list(score=NULL, type=NULL, size_tar=NULL, size_ntar=NULL))
names(source_data_ls) = sub_source
for (id in 1:size_length) {
  sub_source_iter = sub_source[id]
  print(sub_source_iter)
  source_id_output = compute_swlda_score(
    local_use, sub_source_iter, data_type_trn, weight_name_source, size_source[id]
  )
  source_data_ls[[sub_source_iter]]$score = source_id_output$score
  source_data_ls[[sub_source_iter]]$type = source_id_output$type
  source_data_ls[[sub_source_iter]]$size_tar = sum(source_id_output$type == 1)
  source_data_ls[[sub_source_iter]]$size_ntar = sum(source_id_output$type != 1)
}

# initialize the parameters
mu_new_ntar_init = rnorm(1, mean=0, sd=1)
Delta_0_new_init = rnorm(1, mean=0, sd=0.5)
sigmq_sq_new_init = min(rinvgamma(1, shape=2.5, scale=1), 1)
mu_new_tar_init = rtruncnorm(1, a=mu_new_ntar_init, b=Inf, mean=mu_new_ntar_init, sd=1)

param_new = list(
  mu_ntar= mu_new_ntar_init, 
  mu_tar = mu_new_tar_init + Delta_0_new_init,
  Delta_0 = Delta_0_new_init,
  sigma_sq = sigmq_sq_new_init
)
param_source_ls = lapply(1:size_length, function(x) list(mu_tar=NULL, mu_ntar=NULL, sigma_sq=NULL))
names(param_source_ls) = sub_source
delta_select_ls = lapply(1:size_length, function(x) NULL)
names(delta_select_ls) = sub_source
varphi_select_ls = lapply(1:size_length, function(x) NULL)
names(varphi_select_ls) = sub_source

for (id in 1:size_length) {
  sub_source_iter = sub_source[id]
  mu_source_ntar_init = rnorm(1, mean=0, sd=1)
  Delta_0_source_init = rnorm(1, mean=0, sd=0.5)
  sigma_sq_source_init = min(rinvgamma(1, shape=2.5, scale=1), 1)
  mu_source_tar_init = rtruncnorm(1, a=mu_source_ntar_init, b=Inf, mean=mu_source_ntar_init, sd=1)
  
  param_source_ls[[sub_source_iter]]$mu_ntar = mu_source_ntar_init
  param_source_ls[[sub_source_iter]]$mu_tar = mu_source_tar_init + Delta_0_source_init
  param_source_ls[[sub_source_iter]]$Delta_0 = Delta_0_source_init
  param_source_ls[[sub_source_iter]]$sigma_sq = sigma_sq_source_init
  delta_select_ls[[sub_source_iter]] = rbinom(1, 1, 0.5)
  varphi_select_ls[[sub_source_iter]] = 0.5
}

mcmc_num = 5000

for (mcmc_iter in 2:(mcmc_num+1)) {
  if (!mcmc_iter %% 100) {
    print(mcmc_iter)
    # print(param_source_ls$K106$mu_tar[mcmc_iter-1])
    # print(param_source_ls$K106$mu_ntar[mcmc_iter-1])
    # print(param_source_ls$K106$sigma_sq[mcmc_iter-1])
    print(param_new$mu_tar[mcmc_iter-1])
    print(param_new$mu_ntar[mcmc_iter-1])
    print(param_new$sigma_sq[mcmc_iter-1])
  }
  # update mu_source_ntar and mu_source_tar for sub_source_iter
  for (id in 1:size_length) {
    sub_source_iter = sub_source[id]
    mu_source_ntar_iter = update_mu_source_ntar(
      param_source_ls[[sub_source_iter]]$sigma_sq[mcmc_iter-1], delta_select_ls[[sub_source_iter]][mcmc_iter-1],
      source_data_ls[[sub_source_iter]]$score, source_data_ls[[sub_source_iter]]$type,
      source_data_ls[[sub_source_iter]]$size_ntar
    )
    param_source_ls[[sub_source_iter]]$mu_ntar = c(
      param_source_ls[[sub_source_iter]]$mu_ntar, mu_source_ntar_iter
    )
    
    mu_source_tar_iter = update_mu_source_tar(
      param_source_ls[[sub_source_iter]]$mu_ntar[mcmc_iter], 
      param_source_ls[[sub_source_iter]]$Delta_0[mcmc_iter-1],
      param_source_ls[[sub_source_iter]]$sigma_sq[mcmc_iter-1],
      delta_select_ls[[sub_source_iter]][mcmc_iter-1],
      source_data_ls[[sub_source_iter]]$score, source_data_ls[[sub_source_iter]]$type,
      source_data_ls[[sub_source_iter]]$size_tar
    )
    param_source_ls[[sub_source_iter]]$mu_tar = c(
      param_source_ls[[sub_source_iter]]$mu_tar, mu_source_tar_iter
    )
    
    Delta_0_source_output_iter = perform_Delta_0_random_walk(
      mu_source_tar_iter, mu_source_ntar_iter, param_source_ls[[sub_source_iter]]$Delta_0[mcmc_iter-1]
    )
    param_source_ls[[sub_source_iter]]$Delta_0 = c(
      param_source_ls[[sub_source_iter]]$Delta_0, Delta_0_source_output_iter$Delta_0
    )
    # sigma_sq_source_iter = update_sigma_sq_source(
    #   param_source_ls[[sub_source_iter]]$mu_tar[mcmc_iter],
    #   param_source_ls[[sub_source_iter]]$mu_ntar[mcmc_iter],
    #   delta_select_ls[[sub_source_iter]][mcmc_iter-1],
    #   source_data_ls[[sub_source_iter]]$score, source_data_ls[[sub_source_iter]]$type,
    #   source_data_ls[[sub_source_iter]]$size_tar, source_data_ls[[sub_source_iter]]$size_ntar,
    #   a_source=2.5, b_source=1
    # )
    sigma_sq_source_iter = perform_sigma_sq_source_indep_MH(
      param_source_ls[[sub_source_iter]]$mu_tar[mcmc_iter],
      param_source_ls[[sub_source_iter]]$mu_ntar[mcmc_iter],
      param_source_ls[[sub_source_iter]]$sigma_sq[mcmc_iter-1],
      delta_select_ls[[sub_source_iter]][mcmc_iter-1],
      source_data_ls[[sub_source_iter]]$score, 
      source_data_ls[[sub_source_iter]]$type
    )
    # sigma_sq_source_iter = update_sigma_sq_source_empirical(
    #   param_source_ls[[sub_source_iter]]$mu_tar[mcmc_iter],
    #   param_source_ls[[sub_source_iter]]$mu_ntar[mcmc_iter],
    #   source_data_ls[[sub_source_iter]]$score, source_data_ls[[sub_source_iter]]$type,
    #   source_data_ls[[sub_source_iter]]$size_tar, source_data_ls[[sub_source_iter]]$size_ntar
    # )
    param_source_ls[[sub_source_iter]]$sigma_sq = c(
      param_source_ls[[sub_source_iter]]$sigma_sq, sigma_sq_source_iter
    )
  }
  mu_new_ntar_iter = update_mu_new_ntar(
    param_new$sigma_sq[mcmc_iter-1], delta_select_ls, source_data_ls, new_data_ls, sub_source, mcmc_iter-1
  )
  param_new$mu_ntar = c(param_new$mu_ntar, mu_new_ntar_iter)
  
  mu_new_tar_iter = update_mu_new_tar(
    param_new$mu_ntar[mcmc_iter], param_new$Delta_0[mcmc_iter-1],
    param_new$sigma_sq[mcmc_iter-1], delta_select_ls, 
    source_data_ls, new_data_ls, sub_source, mcmc_iter-1
  )
  param_new$mu_tar = c(param_new$mu_tar, mu_new_tar_iter)
  
  Delta_0_new_output_iter = perform_Delta_0_random_walk(
    mu_new_tar_iter, mu_new_ntar_iter, param_new$Delta_0[mcmc_iter-1]
  )
  param_new$Delta_0 = c(param_new$Delta_0, Delta_0_new_output_iter$Delta_0)
  # sigma_sq_new_iter = update_sigma_sq_new(
  #   param_new$mu_tar[mcmc_iter], param_new$mu_ntar[mcmc_iter],
  #   delta_select_ls, source_data_ls, new_data_ls,
  #   sub_source, mcmc_iter-1, a_new=2.5, b_new=1
  # )
  sigma_sq_new_iter = perform_sigma_sq_new_indep_MH(
    param_new$mu_tar[mcmc_iter], param_new$mu_ntar[mcmc_iter],
    param_new$sigma_sq[mcmc_iter-1], delta_select_ls, mcmc_iter-1,
    source_data_ls, new_data_ls, sub_source
  )
  # sigma_sq_new_iter = rinvgamma(1, shape=2.5, scale=1)
  param_new$sigma_sq = c(param_new$sigma_sq, sigma_sq_new_iter)
  
  for (id in 1:size_length) {
    sub_source_iter = sub_source[id]
    delta_select_output = update_delta_select(
      param_source_ls[[sub_source_iter]]$mu_tar[mcmc_iter], 
      param_source_ls[[sub_source_iter]]$mu_ntar[mcmc_iter],
      param_source_ls[[sub_source_iter]]$sigma_sq[mcmc_iter],
      param_new$mu_tar[mcmc_iter], param_new$mu_ntar[mcmc_iter], param_new$sigma_sq[mcmc_iter],
      source_data_ls[[sub_source_iter]]$score, source_data_ls[[sub_source_iter]]$type, phi_0=0.5
    )
    delta_select_iter = delta_select_output$delta_select
    varphi_iter = delta_select_output$phi_post
    
    delta_select_ls[[sub_source_iter]] = c(delta_select_ls[[sub_source_iter]], delta_select_iter)
    # delta_select_ls[[sub_source_iter]] = c(delta_select_ls[[sub_source_iter]], rbinom(1, 1, 0.5))
    varphi_select_ls[[sub_source_iter]] = c(varphi_select_ls[[sub_source_iter]], varphi_iter)
  }
}

for (source_name_iter in sub_source) {
  param_source_ls[[source_name_iter]]$mu_tar = param_source_ls[[source_name_iter]]$mu_tar[2501:5000]
  param_source_ls[[source_name_iter]]$mu_ntar = param_source_ls[[source_name_iter]]$mu_ntar[2501:5000]
  param_source_ls[[source_name_iter]]$sigma_sq = param_source_ls[[source_name_iter]]$sigma_sq[2501:5000]
  param_source_ls[[source_name_iter]]$Delta_0 = param_source_ls[[source_name_iter]]$Delta_0[2501:5000]
  delta_select_ls[[source_name_iter]] = delta_select_ls[[source_name_iter]][2501:5000]
  varphi_select_ls[[source_name_iter]] = varphi_select_ls[[source_name_iter]][2501:5000]
}
param_new$mu_tar = param_new$mu_tar[2501:5000]
param_new$mu_ntar = param_new$mu_ntar[2501:5000]
param_new$sigma_sq = param_new$sigma_sq[2501:5000]
param_new$Delta_0 = param_new$Delta_0[2501:5000]


lapply(1:size_length, function(x) mean(param_source_ls[[x]]$mu_tar))
lapply(1:size_length, function(x) mean(param_source_ls[[x]]$mu_ntar))
# lapply(1:size_length, function(x) mean(param_source_ls[[x]]$sigma_sq[2501:5000]))
lapply(1:size_length, function(x) mean(param_source_ls[[x]]$Delta_0))
# lapply(1:size_length, function(x) mean(delta_select_ls[[x]]))
# lapply(1:size_length, function(x) mean(varphi_select_ls[[x]]))
# 
# mean(param_new$mu_tar[2501:5000])
# mean(param_new$mu_ntar[2501:5000])
# mean(param_new$sigma_sq[2501:5000])

dir.create(file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output'))
file_suffix = paste('letter_num_', letter_num_new, sep='')
stan_output_dir_1 = file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output', 
                              paste('mcmc_summary_', file_suffix, '.pdf', sep=''))

pdf(stan_output_dir_1, width=10, height=10)
par(mfrow=c(4, 4))
for(i in 1:size_length) {
  plot(density(param_source_ls[[i]]$mu_tar), main=paste(sub_source[i], 'mu_tar', sep=', '))
  plot(density(param_source_ls[[i]]$mu_ntar), main=paste(sub_source[i], 'mu_ntar', sep=', '))
  plot(density(param_source_ls[[i]]$sigma_sq), main=paste(sub_source[i], 'sigma_sq', sep=', '))
  hist(delta_select_ls[[i]], main=paste(sub_source[i], 'delta_select', sep=', '), xlim=c(-0.1,1.1))
}

par(mfrow=c(3, 1))
plot(density(param_new$mu_tar), main=paste(sub_new, 'new_mu_tar', sep=', '))
plot(density(param_new$mu_ntar), main=paste(sub_new, 'new_mu_ntar', sep=', '))
plot(density(param_new$sigma_sq), main=paste(sub_new, 'new_sigma_sq', sep=', '))
dev.off()

# sink(file=stan_output_dir_1)
# print(stan_model_fit)
# sink()

stan_output_dir_2 = file.path(parent_eeg_data_dir, data_type_trn, sub_new, 'stan_output', 
                              paste('mcmc_summary_', file_suffix, '.rds', sep=''))
stan_output_ls = list(
  source = param_source_ls,
  delta = delta_select_ls,
  varphi = varphi_select_ls,
  new = param_new
)
saveRDS(stan_output_ls, stan_output_dir_2)




