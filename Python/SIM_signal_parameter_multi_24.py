from self_py_fun.SimGenFun import *
plt.style.use("bmh")

parent_path_sim_dir  = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/SIM_files'.format(parent_path_sim_dir)

E = 2
plot_sim_param_bool = False
plot_sim_data_bool = True

N = 24
K = N
iter_num = 0
rho_val = 0.85
eta_val = 0.0

# create a new directory to save simulated data
scenario_name = 'N_{}_K_{}_multi_xdawn_eeg'.format(N, K)
scenario_name_dir = '{}/{}'.format(parent_data_dir, scenario_name)
if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)
scenario_iter_name_dir = '{}/iter_{}'.format(scenario_name_dir, iter_num)
if not os.path.exists(scenario_iter_name_dir):
    os.mkdir(scenario_iter_name_dir)

# import mcmc files from the real data analysis
temp_dir = '{}/K151_mcmc_samples.mat'.format(parent_data_dir)
temp_df = sio.loadmat(temp_dir, simplify_cells=True)
temp_df = temp_df['chain_1']
B_tar_mcmc = temp_df['B_tar']
B_0_ntar_mcmc = temp_df['B_0_ntar']
sigma_mcmc = temp_df['sigma']
z_mcmc = temp_df['z_vector']

# without conditioning on Z
B_tar_mean = np.mean(B_tar_mcmc, axis=0)
B_0_ntar_mean = np.mean(B_0_ntar_mcmc, axis=0)
sigma_mean = np.round(np.mean(sigma_mcmc, axis=0), decimals=2) * 2.0
z_mean = np.mean(z_mcmc, axis=0)

# output_dir = '{}/numpyro_output'.format(parent_dir)
label_N_dict = {}
for n in range(N):
    subject_n_arr = np.zeros([N])
    subject_n_arr[n] = 1
    label_N_dict['subject_{}'.format(n)] = subject_n_arr

param_true_dict = {
    'target_char': target_char,
    'N': N,
    'E': E,
    'seq_size': [seq_source_size for i in range(N)],
    'signal_length': 25,
    'label_N': label_N_dict,
    'z': z_mean
}
for group_iter in range(N):
    param_true_group_iter_dict = {
        'beta_tar': B_tar_mean[group_iter, ...],
        'beta_ntar': B_0_ntar_mean,
        'sigma': sigma_mean[group_iter, :],
        'rho': rho_val,
        'eta': eta_val
    }
    param_true_dict['group_{}'.format(group_iter)] = param_true_group_iter_dict

target_char_test = list('THE0QUICK0BROWN0FOX')
seq_size_test = 10

if plot_sim_param_bool:
    # visualize the true parameters
    for k in range(K):
        fig_k, ax_k = plt.subplots(1, ncols=E, figsize=(10, 8))
        group_k_name = 'group_{}'.format(k)
        for e in range(E):
            ax_k[e].plot(
                np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_tar'][e, :],
                label='target', color='red'
            )
            ax_k[e].plot(
                np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_ntar'][e, :],
                label='non-target', color='blue'
            )
            ax_k[e].legend(loc='lower right')
            ax_k[e].set_xlabel('Time (unit)')
            ax_k[e].set_ylim((-1.5, 1.5))
            ax_k[e].set_title('{}, channel_{}, \nsigma={:.2f}'.format(
                group_k_name, e + 1, sigma_mean[k, e])
            )
        fig_k.savefig('{}/N_{}_K_{}_multi_xdawn_eeg_true_ERP_functions_group_{}.png'.format(
            scenario_name_dir, N, K, k)
        )
        plt.close()


print('iteration {}'.format(iter_num))
np.random.seed(iter_num)
scenario_name_dir_try = '{}/iter_{}'.format(scenario_name_dir, iter_num)
if not os.path.exists(scenario_name_dir_try):
    os.mkdir(scenario_name_dir_try)

# Multi-channel simulation data
sim_data_dict = generate_simulated_data_multi(
    param_true_dict, scenario_name_dir_try, seed_num=iter_num
)

# examine the simulated training data
if plot_sim_data_bool:
    scenario_name_dir_try_2 = '{}/data_summary_figures'.format(scenario_name_dir_try)
    if not os.path.exists(scenario_name_dir_try_2):
        os.mkdir(scenario_name_dir_try_2)
    for i in range(N):
        subject_i_name = 'subject_{}'.format(i)
        sim_data_dict[subject_i_name]['X'] = np.squeeze(np.array([sim_data_dict[subject_i_name]['X']]), axis=0)
        sim_data_dict[subject_i_name]['Y'] = np.array(sim_data_dict[subject_i_name]['Y'])
        mean_tar_i = np.mean(sim_data_dict[subject_i_name]['X'][sim_data_dict[subject_i_name]['Y'] == 1.0, ...], axis=0)
        mean_ntar_i = np.mean(sim_data_dict[subject_i_name]['X'][sim_data_dict[subject_i_name]['Y'] != 1.0, ...], axis=0)
        fig1, ax1 = plt.subplots(1, E, figsize=(12, 6))
        for e in range(E):
            ax1[e].plot(
                np.arange(param_true_dict['signal_length']), mean_tar_i[e, :], label='target', color='red'
            )
            ax1[e].plot(
                np.arange(param_true_dict['signal_length']), mean_ntar_i[e, :], label='non-target', color='blue'
            )
            ax1[e].legend(loc='lower right')
            ax1[e].set_xlabel('Time (unit)')
            ax1[e].set_ylim([-2, 2])
            ax1[e].set_title('{}, cluster_{}, channel_{}'.format(
                subject_i_name, sim_data_dict[subject_i_name]['label'], e + 1)
            )
            fig1.savefig('{}/plot_sim_data_mean_{}.png'.format(scenario_name_dir_try_2, subject_i_name))

# generate the testing data
sim_data_test_dict = generate_simulated_test_data_multi(
    param_true_dict, sim_data_dict['subject_0']['label'], target_char_test,
    seq_size_test, scenario_name_dir_try, seed_num=iter_num
)


