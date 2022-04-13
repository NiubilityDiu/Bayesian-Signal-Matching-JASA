import sys
import matplotlib.pyplot as plt
from self_py_fun.SimMCMCFun import *
plt.style.use("bmh")
assert numpyro.__version__.startswith("0.9.0")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
target_char = 'T'

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N = 4
    K = 3
    prob_0_option_id = 1
    rho_val = 0.5
    lambda_val = 0.3
    seq_i = 9
    channel_dim = 2

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    rho_val = float(sys.argv[5])
    lambda_val = float(sys.argv[6])
    seq_i = int(sys.argv[7])  # take values from 9, 19, 29, 39, and 49.
    channel_dim = int(sys.argv[8])

signal_length = 25
seq_size_ls = [50 for i in range(N)]

# Scenario 2 with multi-channel data
if K == 3:
    scenario_name = 'N_{}_K_{}_K114_K123_K151_based_option_{}_rho_{}_lambda_{}'.format(
        N, K, prob_0_option_id, rho_val, lambda_val
    )
else:
    scenario_name = 'N_{}_K_{}_K114_K123_K151_K183_based_option_{}_rho_{}_lambda_{}'.format(
        N, K, prob_0_option_id, rho_val, lambda_val
    )

scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)

with open(sim_data_dir, 'r') as file:
    sim_data_dict = json.load(file)

for i in range(N):
    subject_i_name = 'subject_{}'.format(i)
    sim_data_dict[subject_i_name]['X'] = np.reshape(
        np.array(sim_data_dict[subject_i_name]['X']),
        [seq_size_ls[i] * rcp_unit_flash_num, channel_dim, signal_length]
    )
    print(sim_data_dict[subject_i_name]['X'].shape)
    sim_data_dict[subject_i_name]['Y'] = np.reshape(
        np.array(sim_data_dict[subject_i_name]['Y']),
        [seq_size_ls[i] * rcp_unit_flash_num]
    )
    print(sim_data_dict[subject_i_name]['Y'].shape)

input_data = {}
for n in range(N):
    subject_n_name = 'subject_{}'.format(n)
    sim_data_subject_n = sim_data_dict[subject_n_name]
    flash_size_tar_n = int(seq_size_ls[n] * rcp_unit_flash_num / 6)
    flash_size_ntar_n = 5 * flash_size_tar_n
    input_data[subject_n_name] = {
        'target': np.reshape(
            sim_data_subject_n['X'][sim_data_subject_n['Y'] == 1, ...],
            [flash_size_tar_n, channel_dim * signal_length]
        ),
        'non-target': np.reshape(
            sim_data_subject_n['X'][sim_data_subject_n['Y'] == 0, ...],
            [flash_size_ntar_n, channel_dim * signal_length]
        )
    }
# subset the subject 0
input_data['subject_0']['target'] = np.copy(input_data['subject_0']['target'][:(seq_i+1) * 2, :])
input_data['subject_0']['non-target'] = np.copy(input_data['subject_0']['non-target'][:(seq_i+1) * 10, :])


# kernel hyper-parameter
index_x = (np.arange(25) - 12) / 25
kernel_gram_mat_ntar = kernel_gamma_exp(
    index_x, index_x, var=1.0, length=0.4, gamma_val=1.1, noise=0
)
eigen_val_ntar, eigen_fun_ntar = kernel_mercer_representation(kernel_gram_mat_ntar)
print(eigen_val_ntar)

kernel_gram_mat_tar = kernel_gamma_exp(
    index_x, index_x, var=1.0, length=0.3, gamma_val=1.1, noise=0
)
eigen_val_tar, eigen_fun_tar = kernel_mercer_representation(kernel_gram_mat_tar)
print(eigen_val_tar)

eigen_val_dict = {}
eigen_fun_mat_dict = {}
for k in range(K):
    group_k_name = 'group_{}'.format(k)
    eigen_val_dict[group_k_name] = {'target': eigen_val_tar, 'non-target': eigen_val_ntar}
    eigen_fun_mat_dict[group_k_name] = {'target': eigen_fun_tar, 'non-target': eigen_fun_ntar}


# fast fit of the new subject to obtain the reference panel.
nuts_kernel_new = NUTS(signal_new_sim_multi)
mcmc_new = MCMC(nuts_kernel_new, num_warmup=400, num_samples=200)
rng_key_new = random.PRNGKey(0)
mcmc_new.run(
    rng_key_new, input_points=index_x,
    eigen_val_dict=eigen_val_dict['group_0'],
    eigen_fun_mat_dict=eigen_fun_mat_dict['group_0'],
    input_data=input_data['subject_0'], channel_dim=channel_dim, extra_fields=('potential_energy',),
)

# save mcmc dict for reference fit (new subject data only)
mcmc_new_dict = mcmc_new.get_samples()
mcmc_new_dict_dir = '{}/mcmc_new_seq_size_{}.mat'.format(scenario_name_dir, seq_i+1)
sio.savemat(mcmc_new_dict_dir, mcmc_new_dict)

# save mcmc_new summary using sys.stdout
summary_new_dict_dir = '{}/summary_new_seq_size_{}.txt'.format(scenario_name_dir, seq_i+1)
stdoutOrigin = sys.stdout
sys.stdout = open(summary_new_dict_dir, "w")
mcmc_new.print_summary()
sys.stdout.close()
sys.stdout = stdoutOrigin


'''
# load mcmc_dict
mcmc_new_dict_dir = '{}/mcmc_new_seq_size_{}.mat'.format(scenario_name_dir, seq_i + 1)
mcmc_new_dict = sio.loadmat(mcmc_new_dict_dir)
'''

# extract beta summary with new data only and produce the plot for review
beta_tar_new_dict, beta_ntar_new_dict = signal_beta_multi_summary(
    mcmc_new_dict, eigen_fun_mat_dict['group_0'], 'new', channel_dim
)
beta_tar_new_mean = beta_tar_new_dict['beta_tar_mean']
beta_ntar_new_mean = beta_ntar_new_dict['beta_ntar_mean']

fig1, ax1 = plt.subplots(1, 2, figsize=(10, 6))
x_time = np.arange(signal_length)
for channel_e in range(channel_dim):
    # ax1_row = int(channel_e / 3)
    # ax1_col = channel_e % 3
    ax1[channel_e].plot(
        x_time, beta_ntar_new_dict['beta_ntar_mean'][channel_e, :], label='non-target', color='blue'
    )
    ax1[channel_e].plot(
        x_time, beta_tar_new_dict['beta_tar_mean'][channel_e, :], label='target', color='red'
    )
    ax1[channel_e].fill_between(
        x_time, beta_ntar_new_dict['beta_ntar_low'][channel_e, :],
        beta_ntar_new_dict['beta_ntar_upp'][channel_e, :],
        alpha=0.2, label='non-target', color='blue'
    )
    ax1[channel_e].fill_between(
        x_time, beta_tar_new_dict['beta_tar_low'][channel_e, :],
        beta_tar_new_dict['beta_tar_upp'][channel_e, :],
        alpha=0.2, label='target', color='red'
    )
    ax1[channel_e].legend(loc='best')
    ax1[channel_e].set_xlabel('Time (unit)')
    ax1[channel_e].set_title('New subject, cluster_{}, channel_{}'.format(
        sim_data_dict['subject_0']['label'], channel_e+1)
    )
    fig1.savefig('{}/plot_new_seq_size_{}.png'.format(scenario_name_dir, seq_i+1))


# clustering algorithm
mcmc_summary_init_dict = None
mcmc_continue_bool = True
rng_key_init = 1
nuts_kernel = NUTS(signal_integration_sim_multi)
mcmc_iter_dict = {}
while mcmc_continue_bool and rng_key_init <= 5:
    if rng_key_init == 1:
        num_warmup = 400
        num_samples = 200
    else:
        num_warmup = 0
        num_samples = 150
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    rng_key_iter = random.PRNGKey(rng_key_init)
    mcmc.run(
        rng_key_iter, N, K, input_points=index_x,
        eigen_val_dict=eigen_val_dict, eigen_fun_mat_dict=eigen_fun_mat_dict,
        input_data=input_data, channel_dim=channel_dim,
        extra_fields=('potential_energy',), init_params=mcmc_summary_init_dict
    )
    mcmc_iter_dict = mcmc.get_samples()

    # produce the beta function plots
    beta_tar_iter_dict, beta_ntar_iter_dict = signal_integration_beta_multi_summary(
        mcmc_iter_dict, K, eigen_fun_mat_dict, channel_dim
    )

    # measure the beta from clustering and beta from fast fit
    new_diff_var = []
    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        beta_tar_k_mean = beta_tar_iter_dict[group_k_name]['beta_tar_mean']  # (channel_dim, signal_length)
        beta_ntar_k_mean = beta_ntar_iter_dict[group_k_name]['beta_ntar_mean']
        beta_tar_k_diff = beta_tar_k_mean - beta_tar_new_mean
        beta_ntar_k_diff = beta_ntar_k_mean - beta_ntar_new_mean
        beta_k_new_diff_var = np.var(beta_tar_k_diff) + np.var(beta_ntar_k_diff)
        new_diff_var.append(beta_k_new_diff_var)

    new_diff_var = np.array(new_diff_var)
    print('new_diff_var = {}'.format(new_diff_var))

    arg_min_new_diff_var = new_diff_var.argmin()
    if arg_min_new_diff_var != 0:
        mcmc_continue_bool = True
        rng_key_init = rng_key_init + 1
        # resolve label switching issue by switching the labels between actual minimum index and 0.
        mcmc_summary_init_dict = signal_integration_swap_multi(mcmc_iter_dict, N, K, channel_dim, arg_min_new_diff_var)
        print('We need another round of MCMC samples')
    else:
        mcmc_continue_bool = False
        print('MCMC stops.')


# save mcmc_dict
mcmc_iter_dict_dir = '{}/mcmc_seq_size_{}.mat'.format(scenario_name_dir, seq_i + 1)
sio.savemat(mcmc_iter_dict_dir, mcmc_iter_dict)

'''
# load mcmc_dict
mcmc_iter_dict_dir = '{}/mcmc_seq_size_{}.mat'.format(scenario_name_dir, seq_i + 1)
mcmc_iter_dict = sio.loadmat(mcmc_iter_dict_dir)
'''

# save mcmc_new summary using sys.stdout
summary_dict_dir = '{}/summary_seq_size_{}.txt'.format(scenario_name_dir, seq_i+1)
stdoutOrigin = sys.stdout
sys.stdout = open(summary_dict_dir, "w")
mcmc.print_summary()
sys.stdout.close()
sys.stdout = stdoutOrigin

# produce the beta function plots
beta_tar_summary_dict, beta_ntar_summary_dict = signal_integration_beta_multi_summary(
    mcmc_iter_dict, K, eigen_fun_mat_dict, channel_dim
)

for k in range(K):
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 6))
    group_k_name = 'group_{}'.format(k)
    for channel_e in range(channel_dim):
        # ax1_row = int(channel_e / 3)
        # ax1_col = channel_e % 3
        ax1[channel_e].plot(
            x_time, beta_ntar_summary_dict[group_k_name]['beta_ntar_mean'][channel_e, :],
            label='non-target', color='blue'
        )
        ax1[channel_e].plot(
            x_time, beta_tar_summary_dict[group_k_name]['beta_tar_mean'][channel_e, :],
            label='target', color='red'
        )
        ax1[channel_e].fill_between(
            x_time, beta_ntar_summary_dict[group_k_name]['beta_ntar_low'][channel_e, :],
            beta_ntar_summary_dict[group_k_name]['beta_ntar_upp'][channel_e, :],
            alpha=0.2, label='non-target', color='blue'
        )
        ax1[channel_e].fill_between(
            x_time, beta_tar_summary_dict[group_k_name]['beta_tar_low'][channel_e, :],
            beta_tar_summary_dict[group_k_name]['beta_tar_upp'][channel_e, :],
            alpha=0.2, label='target', color='red'
        )
        ax1[channel_e].legend(loc='best')
        ax1[channel_e].set_xlabel('Time (unit)')
        ax1[channel_e].set_title('Cluster_{}, channel_{}'.format(
            k, channel_e+1)
        )
    fig1.savefig('{}/plot_seq_size_{}_group_{}.png'.format(scenario_name_dir, seq_i+1, k))
