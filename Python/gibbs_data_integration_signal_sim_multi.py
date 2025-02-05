from self_py_fun.MCMCMultiFun import *
import tqdm
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
step_size_adjust_bool = False
E = 2
mcmc_bool = True
plot_gibbs_bool = True
xdawn_bool, letter_dim_sub = False, target_char_size


if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    sim_erp_multi_dir = '{}/SIM_files/Chapter_3/sim_erp_multi.mat'.format(parent_path_local)
    iter_num = 0
    N_total = 7
    K = 3
    prob_0_option_id = 2
    seq_i = 4

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    sim_erp_multi_dir = '{}/SIM_files/Chapter_3/sim_erp_multi.mat'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N_total = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    seq_i = int(sys.argv[5])  # take values from 0 to 9

# np.random.seed(iter_num)
seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_multi_option_{}'.format(N_total, K, prob_0_option_id)
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N_total, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)

cluster_name_dir = '{}/borrow_gibbs'.format(scenario_name_dir)
if not os.path.exists(cluster_name_dir):
    os.mkdir(cluster_name_dir)

ref_name_dir = '{}/reference_numpyro'.format(scenario_name_dir)
if not os.path.exists(ref_name_dir):
    os.mkdir(ref_name_dir)


# create the kernel matrix and related hyper-parameters
length_ls_2 = [[0.3, 0.2]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']


# other hyper-parameters:
psi_loc, psi_scale = 0.0, 1.0
delta_mean = 3.0
delta_sd = 1.0
sigma_loc = delta_mean**2/delta_sd**2+2
sigma_scale = delta_mean * (delta_mean**2 / delta_sd**2 + 1)

eta_grid = np.round(np.arange(0, 1, 0.05), 2).tolist()
rho_grid = np.round(np.arange(0, 1, 0.05), 2).tolist()

# three chains
parameter_name_ls = ['B_tar', 'B_0_ntar',
                     'A_tar', 'A_0_ntar',
                     'psi_tar', 'psi_0_ntar',
                     'sigma', 'rho', 'eta',
                     'z_vector']

rho_name_id = parameter_name_ls.index('rho')
eta_name_id = parameter_name_ls.index('eta')

mcmc_summary_merge_dict = {}
for name_id, name_iter in enumerate(parameter_name_ls):
    mcmc_summary_merge_dict[name_iter] = []
mcmc_summary_merge_dict['log_joint_prob'] = []
num_chain = 2
num_burnin = 5000
num_samples = 1000
iter_check_val = 100

# import sim_data
# approx_threshold = 1.0
source_data, new_data = import_sim_data(
    sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length
)

if mcmc_bool:

    for chain_id in range(num_chain):
        print('\n chain id: {}\n'.format(chain_id + 1))
        estimate_rho_bool = True
        estimate_eta_bool = True
        log_joint_prob_iter = 0

        # semi-supervised clustering problem
        mcmc_summary_dict = initialize_data_multi_fast_calculation_sim(N_total, E, signal_length, seq_i, seq_size_ls[0],
                                                                       new_data, eigen_fun_mat_dict, eigen_val_dict,
                                                                       scenario_name_dir)
        parameter_iter_ls = [np.copy(mcmc_summary_dict[name_iter][0]) for name_iter in parameter_name_ls]
        # mh_param_size = 3
        step_size_ls = [np.ones([N_total]) * 0.1, np.ones([N_total, E]) * 0.1, 0.1]
        accept_sum_ls = [np.zeros(N_total), np.ones([N_total, E]), 0]

        eta_fix_ls = []
        rho_fix_ls = []
        z_num_vec = np.zeros([N_total-1])

        for iter_id in tqdm.tqdm(range(num_burnin+num_samples)):
            ref_mcmc_id = np.random.random_integers(low=0, high=199)
            parameter_iter_ls, accept_iter_ls, log_joint_prob_iter = update_multi_gibbs_sampler_per_iteration(
                eigen_fun_mat_dict, eigen_val_dict, source_data, new_data, N_total, E, signal_length, None,
                step_size_ls, estimate_eta_bool, estimate_rho_bool,
                ref_name_dir, seq_source_size, seq_i, xdawn_bool, letter_dim_sub,
                *parameter_iter_ls,
                psi_loc=psi_loc, psi_scale=psi_scale, sub_new_name=0,
                sigma_loc=sigma_loc, sigma_scale=sigma_scale,
                eta_grid=eta_grid, rho_grid=rho_grid,
                ref_mcmc_bool=True, ref_mcmc_id=ref_mcmc_id,
                approx_threshold=log_lhd_diff_approx, z_n_match_prior_prob=0.5,
                approx_random_bool=True
            )

            for n in range(N_total):
                accept_sum_ls[0][n] = accept_sum_ls[0][n] + accept_iter_ls[0][n]
                accept_sum_ls[1][n, :] = accept_sum_ls[1][n, :] + accept_iter_ls[1][n, :]
            accept_sum_ls[2] = accept_sum_ls[2] + accept_iter_ls[2]

            z_num_vec = z_num_vec + parameter_iter_ls[-1]

            if iter_id % iter_check_val == 0:
                print('MCMC iteration number: {}'.format(iter_id))
                for name_id, name in enumerate(parameter_name_ls):
                    if name not in ['A_tar', 'A_0_ntar']:
                        print('{}: \n {}'.format(name, parameter_iter_ls[name_id]))
                print('log_joint_prob: \n {} \n'.format(log_joint_prob_iter))

                print('z_vector has mean {}'.format(z_num_vec / iter_check_val))
                # reset z_vector summation
                z_num_vec = np.zeros([N_total-1])

                # check step size and acceptance rate
                if step_size_adjust_bool and 0 < iter_id < int(num_burnin * 0.6):
                    accept_mean_ls = [accept_sum_ls[0] / iter_check_val, accept_sum_ls[1] / iter_check_val,
                                      accept_sum_ls[2] / iter_check_val]
                    for n in range(N_total):
                        step_size_ls[0][n] = adjust_MH_step_size_random_walk(step_size_ls[0][n], accept_mean_ls[0][n])
                        for e_iter in range(E):
                            step_size_ls[1][n, e_iter] = adjust_MH_step_size_random_walk(step_size_ls[1][n, e_iter],
                                                                                         accept_mean_ls[1][n, e_iter])
                    step_size_ls[2] = adjust_MH_step_size_random_walk(step_size_ls[2], accept_mean_ls[2])

                    print(
                        'acceptance rate for psi_tar, sigma, and psi_0_ntar are \n{}, \n{}, and {}, respectively.'.format(
                            accept_mean_ls[0], accept_mean_ls[1].T, accept_mean_ls[2])
                    )
                    print('step_sizes for psi_tar, sigma, and psi_0_ntar are \n{}, \n{}, and {}, respectively.'.format(
                        step_size_ls[0], step_size_ls[1].T, step_size_ls[2])
                    )
                    # reset accept_sum_ls
                    accept_sum_ls = [np.zeros(N_total), np.zeros([N_total, E]), 0]

            if iter_id < int(num_burnin * 0.8):
                eta_fix_ls.append(parameter_iter_ls[eta_name_id])
                rho_fix_ls.append(parameter_iter_ls[rho_name_id])

            if iter_id == int(num_burnin * 0.8):
                eta_fix = stats.mode(np.stack(eta_fix_ls, axis=0), axis=0)[0][0]
                print('eta fix!')
                parameter_iter_ls[eta_name_id] = np.copy(eta_fix)
                estimate_eta_bool = False

                rho_fix = stats.mode(np.stack(rho_fix_ls, axis=0), axis=0)[0][0]
                print('rho fix!')
                parameter_iter_ls[rho_name_id] = np.copy(rho_fix)
                estimate_rho_bool = False

            # record MCMC samples after num_burnin iterations.
            if iter_id > num_burnin:
                for name_id, name in enumerate(parameter_name_ls):
                    mcmc_summary_dict[name].append(np.copy(parameter_iter_ls[name_id]))
                mcmc_summary_dict['log_joint_prob'].append(log_joint_prob_iter)

        for _, name in enumerate(parameter_name_ls):
            mcmc_summary_merge_dict[name].append(mcmc_summary_dict[name])
        mcmc_summary_merge_dict['log_joint_prob'].append(mcmc_summary_dict['log_joint_prob'])

    # save mcmc_dict
    mcmc_summary_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster.mat'.format(cluster_name_dir, seq_i + 1)
    for name_id, name_iter in enumerate(parameter_name_ls):
        mcmc_summary_merge_dict[name_iter] = np.concatenate(mcmc_summary_merge_dict[name_iter], axis=0)
    sio.savemat(mcmc_summary_dict_dir, mcmc_summary_merge_dict)


# load mcmc_dict
mcmc_summary_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster.mat'.format(cluster_name_dir, seq_i + 1)
mcmc_summary_merge_dict = sio.loadmat(mcmc_summary_dict_dir)
color_ls = ['red', 'blue', 'green']

sigma_true_ls = [np.array([8.0, 8.0]), np.array([8.0, 6.0]), np.array([4.0, 2.0])]
rho_true_ls = [0.7, 0.7, 0.5]
eta_true_ls = [0.6, 0.4, 0.4]

if prob_0_option_id == 1:
    subject_true_group_ls = [0, 1, 1, 1, 2, 2, 2]
elif prob_0_option_id == 2:
    subject_true_group_ls = [0, 0, 0, 1, 1, 2, 2]
else:
    subject_true_group_ls = []

if plot_gibbs_bool:
    fig0, ax0 = plt.subplots(4, N_total, figsize=(30, 15))

    for n in range(N_total):
        for chain_id in range(num_chain):
            low_id = chain_id * num_samples
            upp_id = (chain_id + 1) * num_samples
            ax0[0, n].plot(mcmc_summary_merge_dict['rho'][low_id:upp_id, n],
                           label='rho, chain_{}'.format(chain_id + 1), color=color_ls[chain_id])
            ax0[0, n].legend(loc='upper right')
            ax0[0, n].axhline(y=rho_true_ls[subject_true_group_ls[n]], linestyle='--')
            ax0[0, n].set_ylim([0.2, 0.8])
            ax0[0, n].set_title('Participant {}'.format(n + 1))

            ax0[1, n].plot(mcmc_summary_merge_dict['eta'][low_id:upp_id, n],
                           label='eta, chain_{}'.format(chain_id + 1), color=color_ls[chain_id])
            ax0[1, n].legend(loc='upper right')
            ax0[1, n].axhline(y=eta_true_ls[subject_true_group_ls[n]], linestyle='--')
            ax0[1, n].set_ylim([0, 1])
            ax0[1, n].set_title('Participant {}'.format(n + 1))

            for e in range(E):
                ax0[2 + e, n].plot(mcmc_summary_merge_dict['sigma'][low_id:upp_id, n, e],
                                   label='sigma, e_{}, chain_{}'.format(e + 1, chain_id + 1),
                                   color=color_ls[chain_id])
                ax0[2 + e, n].legend(loc='upper right')
                ax0[2 + e, n].axhline(y=sigma_true_ls[subject_true_group_ls[n]][e], linestyle='--')
                ax0[2 + e, n].set_title('Participant {}'.format(n + 1))
                ax0[2 + e, n].set_ylim([0, 10])
    fig0.savefig('{}/trace_plot_seq_size_{}_chain_{}_cluster.png'.format(cluster_name_dir, seq_i + 1, num_chain))

    q_low = 0.025; q_upp = 1 - q_low
    B_tar_mcmc = mcmc_summary_merge_dict['B_tar']
    B_0_ntar_mcmc = mcmc_summary_merge_dict['B_0_ntar']
    B_tar_summary_dict = {
        'mean': np.mean(B_tar_mcmc, axis=0),
        'low': np.quantile(B_tar_mcmc, axis=0, q=q_low),
        'upp': np.quantile(B_tar_mcmc, axis=0, q=q_upp)
    }
    B_0_ntar_summary_dict = {
        'mean': np.mean(B_0_ntar_mcmc, axis=0),
        'low': np.quantile(B_0_ntar_mcmc, axis=0, q=q_low),
        'upp': np.quantile(B_0_ntar_mcmc, axis=0, q=q_upp)
    }

    fig1, ax1 = plt.subplots(E, N_total, figsize=(24, 12))
    x_time = np.arange(signal_length)

    sim_erp_multi = sio.loadmat(sim_erp_multi_dir)
    rcp_function_multi_smooth_dict_1 = {
        'group_0': {'target': sim_erp_multi['group_0'][..., 0].T, 'non-target': sim_erp_multi['group_0'][..., 1].T},
        'group_1': {'target': sim_erp_multi['group_1'][..., 0].T, 'non-target': sim_erp_multi['group_1'][..., 1].T},
        'group_2': {'target': sim_erp_multi['group_2'][..., 0].T, 'non-target': sim_erp_multi['group_2'][..., 1].T}
    }

    for n in range(N_total):
        true_group_n_name = 'group_{}'.format(subject_true_group_ls[n])
        for e in range(E):

            ax1[e, n].plot(x_time, rcp_function_multi_smooth_dict_1[true_group_n_name]['target'][e, :],
                           label='Target (true)', color='red')
            ax1[e, n].fill_between(
                x_time, B_tar_summary_dict['low'][n, e, :],
                B_tar_summary_dict['upp'][n, e, :], alpha=0.2, label='Target', color='red'
            )
            ax1[e, n].set_ylim([-5, 5])
            ax1[e, n].legend(loc='best')
            ax1[e, n].set_xlabel('Time (unit)')
            ax1[e, n].set_title('Participant {}'.format(n))

            if n == 0:
                ax1[e, n].plot(x_time, rcp_function_multi_smooth_dict_1[true_group_n_name]['non-target'][e, :],
                               label='Non-target (true)', color='blue')
                ax1[e, n].fill_between(
                    x_time, B_0_ntar_summary_dict['low'][e, :],
                    B_0_ntar_summary_dict['upp'][e, :], alpha=0.2, label='Non-target', color='blue'
                )
    fig1.savefig('{}/plot_seq_size_{}_{}_chains_cluster.png'.format(cluster_name_dir, seq_i + 1, num_chain))

