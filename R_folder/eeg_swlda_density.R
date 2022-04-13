# rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
# local_use = T
local_use = (args[1] == 'T' || args[1] == 'True')
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
  # letter_num_new = 3
} else {
  parent_dir = '/home/mtianwen'
  parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
  source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
  source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
  rstan_options(auto_write=F)  # only for server running
  sub_new = K_sub_ids[as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))]
  # sub_new = args[2]
  # letter_num_new = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
  # letter_num_new = as.integer(args[2])
}
# total_sub_num = 20
seed_num = 612
parent_eeg_data_dir = file.path(parent_dir, 'EEG_MATLAB_data')
data_type_trn = 'TRN_files'

score_vec = NULL
type_vec = NULL
size_vec = NULL
d_prime_vec = NULL

for (letter_num_new in 1:12) {
  sample_size_new = 12 * 15 * letter_num_new
  weight_name = paste('Weight_001_BCI_TRN_New_Only_Letter_', letter_num_new, sep='')
  new_data_output = compute_swlda_score(
    local_use, sub_new, data_type_trn, weight_name, sample_size_new
  )
  x_new = new_data_output$score
  y_new = new_data_output$type
  
  score_vec = c(score_vec, x_new)
  type_vec = c(type_vec, y_new)
  size_vec = c(size_vec, rep(letter_num_new, sample_size_new))
  
  target_mean_new = mean(x_new[y_new==1])
  ntarget_mean_new = mean(x_new[y_new==-1])
  sd_new = sd(x_new)
  d_prime = (target_mean_new - ntarget_mean_new) / sd_new
  d_prime_vec = c(d_prime_vec, rep(d_prime, sample_size_new))
}

score_df = data.frame(
  score=score_vec, type=type_vec, size=size_vec, dprime=d_prime_vec,
  subtitle = paste('Size=', size_vec, ', dprime=', 
                   signif(d_prime_vec, digits=2), sep='')
)
score_df$subtitle = factor(score_df$subtitle, levels=unique(score_df$subtitle))

p_score_density = ggplot(data=score_df, aes(x=score, color=factor(type, levels=c(1,-1), labels=c('Target', 'Non-target')))) + 
  geom_density() + facet_wrap(~factor(subtitle), nrow=3, ncol=4) +
  # xlim(c(-4,4)) +
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
p_score_density

output_dir = file.path(
  parent_eeg_data_dir, data_type_trn, sub_new, 'swLDA',
  'Weight_001_BCI_TRN_New_Only_Density.png'
)
ggsave(
  output_dir, p_score_density, width=240, height=160, 
  units='mm', dpi=300
)
