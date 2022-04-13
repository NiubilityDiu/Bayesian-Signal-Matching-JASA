rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
local_use = T
# local_use = (args[1] == 'T')
library(ggplot2)
library(gridExtra)
library(R.matlab)
library(dplyr)
library(patchwork)

if (local_use) {
  parent_path = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  r_fun_path = file.path(parent_path, 'Chapter_3', 'R_folder')
  design_num = 183
  dec_factor = 8
  n_length_fit = ifelse(dec_factor == 4, 50, 25)
  } else {
  parent_path = '/home/mtianwen'
  r_fun_path = '/home/mtianwen/Chapter_3/R_folder'
  design_num = as.integer(args[2])
  dec_factor = as.integer(args[3])
}
n_length_fit = ifelse(dec_factor == 4, 50, 25)
bp_low = 0.5
bp_upp = 6
time_length = 25 / 32 * 1000
source(file.path(r_fun_path, 'self_R_fun', 'global_constant.R'))

eeg_signal_source_dir = file.path(
  parent_path, 'EEG_MATLAB_data', 'SIM_files', 'Chapter_3', 'eeg_signal_source'
)
# dir.create(file.path(eeg_signal_source_dir, 'fit_plots'), showWarnings = F)
multi_channel_num = 1:16

# import down-sample eeg data file
eeg_dat_dir = Sys.glob(file.path(eeg_signal_source_dir, 'EEGDataDown', paste('K', design_num, '*', sep='')))
eeg_mat = readMat(eeg_dat_dir)
eeg_signal = eeg_mat$eeg.signals
output_var_name = paste('K', design_num, '_marginal_variance.mat', sep="")
marginal_var = apply(eeg_signal, 2, var)
print(marginal_var)
writeMat(file.path(eeg_signal_source_dir, 'Misc', output_var_name), variance = marginal_var)





# import mcmc file
mcmc_dat_dir = Sys.glob(file.path(eeg_signal_source_dir, 'MCMC', paste('K', design_num, '*', sep='')))
mcmc_mat = readMat(mcmc_dat_dir)

beta_vec_mean = beta_vec_upp = beta_vec_low = channel_vec = type_vec = time_vec =NULL
zeta_vec_median = NULL
# channel_name_num_e = channel_name_short[e]

zeta = mcmc_mat$zeta
zeta_median = apply(zeta, c(2, 3), median)
zeta_vec_median = c(zeta_vec_median, rep(as.vector(t(zeta_median)), 2))
zeta_0 = 0.4

beta_tar = mcmc_mat$beta.tar[, , , 1]
beta_ntar = mcmc_mat$beta.ntar[, , , 1]
# merge raw beta estimates based on zeta_e_median
for (ee in 1:num_electrode) {
  for (ii in 1:n_length_fit) {
    if (zeta_median[ee, ii] <= zeta_0) {
      beta_temp = (beta_tar[, ee, ii] + 5 * beta_ntar[, ee, ii]) / 6
      beta_tar[, ee, ii] = beta_temp
      beta_ntar[, ee, ii] = beta_temp
    }
  }
}
beta_tar_mean = apply(beta_tar, c(2, 3), mean)
beta_ntar_mean = apply(beta_ntar, c(2, 3), mean)

beta_tar_upp = apply(beta_tar, c(2, 3), quantile, 0.95)
beta_tar_low = apply(beta_tar, c(2, 3), quantile, 0.05)

beta_ntar_upp = apply(beta_ntar, c(2, 3), quantile, 0.95)
beta_ntar_low = apply(beta_ntar, c(2, 3), quantile, 0.05)

beta_vec_mean = c(beta_vec_mean, 
                  as.vector(t(beta_tar_mean)), 
                  as.vector(t(beta_ntar_mean)))
beta_vec_low = c(beta_vec_low, 
                 as.vector(t(beta_tar_low)), 
                 as.vector(t(beta_ntar_low)))
beta_vec_upp = c(beta_vec_upp, 
                 as.vector(t(beta_tar_upp)), 
                 as.vector(t(beta_ntar_upp)))

channel_vec = c(channel_vec, rep(rep(multi_channel_num[1:num_electrode], each=n_length_fit), 2))
type_vec = c(type_vec, rep(c('Target', 'Non-target'), each=n_length_fit*num_electrode))
time_vec = c(time_vec, rep(seq(0, time_length, length.out=n_length_fit), 2*num_electrode))

mcmc_multi_dat = data.frame(
  mean = beta_vec_mean,
  low = beta_vec_low,
  upp = beta_vec_upp,
  zeta = zeta_vec_median,
  channel = channel_vec,
  type = type_vec,
  time = time_vec
)
mcmc_multi_dat$type = factor(mcmc_multi_dat$type, levels=c('Target', 'Non-target'))
mcmc_multi_dat$channel = factor(mcmc_multi_dat$channel, levels=1:16, 
                                labels=channel_name_short)
mcmc_multi_dat$zeta_60 = ifelse(mcmc_multi_dat$zeta >= 0.6, 0.6, -1)
mcmc_multi_dat$zeta_75 = ifelse(mcmc_multi_dat$zeta >= 0.75, 0.75, -1)
mcmc_multi_dat$zeta_90 = ifelse(mcmc_multi_dat$zeta >= 0.90, 0.90, -1)

time_upper_lim = 800
y_upper_lim = ceiling(max(mcmc_multi_dat$upp))
y_lower_lim = floor(min(mcmc_multi_dat$low))


p_multi_spatial_ls = lapply(1:num_electrode, function(x) NULL)
names(p_multi_spatial_ls) = channel_name_short
for(ee in 1:num_electrode) {
  cee = channel_name_short[ee]
  p_multi_spatial_ls[[ee]] = ggplot(
    data=mcmc_multi_dat %>% filter(channel==cee), aes(x=time, y=mean)) +
    geom_line(size=1.0, aes(x=time, y=mean, color=type)) +
    # geom_hline(yintercept=0, linetype=2, alpha=0.5) + 
    geom_ribbon(aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.5) +
    geom_point(size=1.0, aes(x=time, y=mean, color=type), shape=1) +
    scale_x_continuous(limits=c(0, time_upper_lim),
                       breaks=seq(0, time_upper_lim, length.out=5)) +
    scale_y_continuous(limits=c(y_lower_lim, y_upper_lim), 
                       breaks=seq(y_lower_lim, y_upper_lim, by=2)) +
    xlab('') + ylab('') + ggtitle(cee) +
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
}
# Rearrange the plots by the spatial distribution of EEG channels
layout_eeg = '
#ABC#
DEFGH
#I#J#
#KLM#
#NOP#
'
p_multi_dat = 
  p_multi_spatial_ls$F3 + p_multi_spatial_ls$Fz + 
  p_multi_spatial_ls$F4 + p_multi_spatial_ls$T7 + 
  p_multi_spatial_ls$C3 + p_multi_spatial_ls$Cz + 
  p_multi_spatial_ls$C4 + p_multi_spatial_ls$T8 + 
  p_multi_spatial_ls$CP3 + p_multi_spatial_ls$CP4 + 
  p_multi_spatial_ls$P3 + p_multi_spatial_ls$Pz + 
  p_multi_spatial_ls$P4 + p_multi_spatial_ls$PO7 + 
  p_multi_spatial_ls$Oz + p_multi_spatial_ls$PO8 +
  plot_layout(design=layout_eeg, guides='collect') & theme(legend.position = 'bottom')
output_dir = paste('K', design_num, '_erp_function', sep="")
ggsave(file.path(eeg_signal_source_dir, 'Figures', paste(output_dir, '.png', sep="")),
       p_multi_dat, width=300, height=250, units='mm', dpi=400)



p_multi_spatial_zeta_ls = lapply(1:num_electrode, function(x) NULL)
names(p_multi_spatial_zeta_ls) = channel_name_short
for (ee in 1:num_electrode) {
  cee = channel_name_short[ee]
  mcmc_multi_zeta_dat = data.frame(
    channel = cee,
    time = rep(seq(0, 25/32*1000, length.out=n_length_fit), 3),
    zeta_binary = c(mcmc_multi_dat$zeta_60[mcmc_multi_dat$type=='Target' & 
                                             mcmc_multi_dat$channel == cee],
                    mcmc_multi_dat$zeta_75[mcmc_multi_dat$type=='Target' & 
                                             mcmc_multi_dat$channel == cee],
                    mcmc_multi_dat$zeta_90[mcmc_multi_dat$type=='Target' & 
                                             mcmc_multi_dat$channel == cee]),
    threshold = rep(c(0.6, 0.75, 0.9), each=n_length_fit)
  )
  mcmc_multi_zeta_dat$threshold = factor(
    mcmc_multi_zeta_dat$threshold, levels=c(0.6, 0.75, 0.9),
    labels=c('60% Confidence', '75% Confidence', '90% Confidence')
  )
  p_multi_spatial_zeta_ls[[ee]] = ggplot(
    data=mcmc_multi_zeta_dat, aes(x=time, y=zeta_binary)) +
    geom_line(size=2.0, aes(x=time, y=zeta_binary, color=threshold)) +
    scale_x_continuous(limits=c(0, time_upper_lim),
                       breaks=seq(0, time_upper_lim, length.out=5)) +
    scale_y_continuous(limits=c(0.5, 1), breaks=seq(0.5, 1, by=0.1)) +
    xlab('') + ylab('') + ggtitle(cee) +
    scale_color_brewer(palette="Accent") +
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
}

# Rearrange the plots by the spatial distribution of EEG channels
layout_eeg = '
#ABC#
DEFGH
#I#J#
#KLM#
#NOP#'

p_multi_zeta_dat = 
  p_multi_spatial_zeta_ls$F3 + p_multi_spatial_zeta_ls$Fz + 
  p_multi_spatial_zeta_ls$F4 + p_multi_spatial_zeta_ls$T7 + 
  p_multi_spatial_zeta_ls$C3 + p_multi_spatial_zeta_ls$Cz + 
  p_multi_spatial_zeta_ls$C4 + p_multi_spatial_zeta_ls$T8 + 
  p_multi_spatial_zeta_ls$CP3 + p_multi_spatial_zeta_ls$CP4 + 
  p_multi_spatial_zeta_ls$P3 + p_multi_spatial_zeta_ls$Pz + 
  p_multi_spatial_zeta_ls$P4 + p_multi_spatial_zeta_ls$PO7 + 
  p_multi_spatial_zeta_ls$Oz + p_multi_spatial_zeta_ls$PO8 +
  plot_layout(design=layout_eeg, guides='collect') & theme(legend.position = 'bottom')
output_dir = paste('K', design_num, '_zeta_split_window', sep="")
ggsave(file.path(eeg_signal_source_dir, 'Figures', paste(output_dir, '.png', sep="")),
       p_multi_zeta_dat, width=300, height=250, units='mm', dpi=400)
  

