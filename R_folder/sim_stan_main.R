# rm(list=ls(all.names=T))
args = commandArgs(trailingOnly = T)
# local_use = T
local_use = (args[1] == 'T' || args[1] == 'True')
library(rstan)
library(R.matlab)

if (local_use) {
  parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  # parent_dir = 'K:\\Dissertation\\Dataset and Rcode'
  # parent_dir = 'C:/Users/mtwen/Downloads'
  rstan_options(auto_write=T)  # only for local running
  seq_i = 1
  seed_num = 2
  set.seed(seed_num)
  
} else {
  parent_dir = '/home/mtianwen/'
  rstan_options(auto_write=F) 
  seq_i = as.integer(args[2])
  seed_num = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
  set.seed(seed_num)
}
sim_name = paste('source_4_new_seq_', seq_i, '_id_', seed_num, sep='')
parent_dir_r = file.path(parent_dir, 'Chapter_3', 'R_folder')
source(file.path(parent_dir_r, 'self_R_fun', 'self_defined_fun.R'))
source(file.path(parent_dir_r, 'self_R_fun', 'global_constant.R'))
parent_dir_sim_data = file.path(parent_dir, 'EEG_MATLAB_data', 'SIM_files', 'Chapter_3')


# Create simulated data
# Parameter candidate set
# Simulate from target character with stimulus-code/-type indicators
target_chars_train = strsplit('THE', '')[[1]]
target_chars_test = strsplit('NIUBILITYDIU', '')[[1]]
seq_num_new_train = 5
seq_num_source_train = 1
seq_num_new_test = 5
param_ls_train = list(mu_tar = 0.5, mu_ntar = -0.2, sd = 0.3)

# Target training set
sim_new_train = generate_simulated_data(target_chars_train, seq_num_new_train, rcp_key_array, param_ls_train)
# Target testing set
sim_new_test = generate_simulated_data(target_chars_test, seq_num_new_test, rcp_key_array, param_ls_train)

# Source training set
param_ls_source_1 = param_ls_train
param_ls_source_2 = list(mu_tar = 0, mu_ntar = -0.7, sd = 0.3)
param_ls_source_3 = list(mu_tar = 1.0, mu_ntar = 0.3, sd = 0.3)
param_ls_source_4 = list(mu_tar = 0.5, mu_ntar = 0, sd = 0.3)
sim_source_1 = generate_simulated_data(target_chars_train, seq_num_source_train, rcp_key_array, param_ls_source_1)
sim_source_2 = generate_simulated_data(target_chars_train, seq_num_source_train, rcp_key_array, param_ls_source_2)
sim_source_3 = generate_simulated_data(target_chars_train, seq_num_source_train, rcp_key_array, param_ls_source_3)
sim_source_4 = generate_simulated_data(target_chars_train, seq_num_source_train, rcp_key_array, param_ls_source_4)

# Combine all source and target datasets
stan_data_input = list(
  size_source = c(sim_source_1$size, sim_source_2$size, sim_source_3$size, sim_source_4$size),
  x_source = c(as.numeric(sim_source_1$signal), as.numeric(sim_source_2$signal),
               as.numeric(sim_source_3$signal), as.numeric(sim_source_4$signal)),
  y_source = c(as.numeric(sim_source_1$type), as.numeric(sim_source_2$type),
               as.numeric(sim_source_3$type), as.numeric(sim_source_4$type)),
  index_source = c(rep(1, sim_source_1$size), rep(2, sim_source_2$size),
                   rep(3, sim_source_3$size), rep(4, sim_source_4$size)),
  x_new = as.numeric(sim_new_train$signal[1:(12 * seq_i),]),
  y_new = as.numeric(sim_new_train$type[1:(12 * seq_i),]),
  sub_n_source = 4,
  total_size_source = sim_source_1$size + sim_source_2$size + 
    sim_source_3$size + sim_source_4$size,
  size_new = 12 * seq_i * length(target_chars_train)
)

stan_output_folder_dir = dir.create(
  file.path(parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''))
)

# save the dataset
data_dir = file.path(
  parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''),
  paste('sim_', sim_name, '_data.rds', sep='')
)
saveRDS(stan_data_input, data_dir)

# fit the stan model
stan_model_obj = stan_model(
  file=Sys.glob(file.path(parent_dir_r, 'stan', 'participant_selection.stan')), model_name='select'
)
stan_model_fit = sampling(
  stan_model_obj, data=stan_data_input, chains=3, seed=seed_num, iter=4000, warmup=3000
)

# save the summary table
stan_output_dir_1 = file.path(
  parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''),
  paste('sim_', sim_name, '_output.txt', sep='')
)
sink(file=stan_output_dir_1)
print(stan_model_fit)
sink()

# save the mcmc samples
stan_output_dir_2 = file.path(
  parent_dir_sim_data, 'Stan_output', paste('sim_source_4_new_seq_', seq_i, sep=''),
  paste('sim_', sim_name, '_output.rds', sep='')
)
stan_model_output = extract(stan_model_fit)
saveRDS(stan_model_output, stan_output_dir_2)
