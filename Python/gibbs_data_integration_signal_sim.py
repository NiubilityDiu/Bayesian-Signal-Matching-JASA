from self_py_fun.MCMCFun import *
import scipy.io as sio
import tqdm
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
step_size_adjust_bool = False
E = 1
source_name_ls = None
mcmc_bool = True


if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N_total = 7
    K = 3
    prob_0_option_id = 2
    sigma_val = 3.0
    rho_val = 0.6
    seq_i = 4

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N_total = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])
    seq_i = int(sys.argv[7])  # take values from 0 to 9

seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_option_{}_sigma_{}_rho_{}'.format(
    N_total, K, prob_0_option_id, sigma_val, rho_val
)

# Scenario 1:
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N_total, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)
cluster_name_dir = '{}/borrow_gibbs'.format(scenario_name_dir, K)
if not os.path.exists(cluster_name_dir):
    os.mkdir(cluster_name_dir)

ref_name_dir = '{}/reference_numpyro'.format(scenario_name_dir)
if not os.path.exists(ref_name_dir):
    os.mkdir(ref_name_dir)

# create the kernel matrix and related hyper-parameters
# make sure kernel hyper-parameters are consistent.
length_ls_2 = [[0.4, 0.3]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']

# alpha_random = mvn(mean=np.zeros_like(eigen_val_dict['target']), cov=np.diag(eigen_val_dict['target'])).rvs(size=10)
# beta_random = eigen_fun_mat_dict['target'][np.newaxis, ...] @ alpha_random[..., np.newaxis]
# plt.plot(np.squeeze(beta_random, axis=-1).T)
# plt.show()

# other hyper-parameters:
psi_loc, psi_scale = 0.0, 1.0
sigma_loc, sigma_scale = 0.0, 0.5
# # sigma_df = 2
# rho_tf_loc = 0.0
# rho_tf_scale = 1.0
rho_grid = np.round(np.arange(0, 1, 0.05), 2).tolist()

# three chains:
parameter_name_ls = ['beta_tar', 'beta_0_ntar', 'alpha_tar', 'alpha_0_ntar', 'psi_tar', 'psi_0_ntar',
                     'sigma', 'rho', 'z_vector', 'z_vector_prob']
rho_name_id = parameter_name_ls.index('rho')
z_vector_name_id = parameter_name_ls.index('z_vector')

mcmc_summary_merge_dict = {}
for name_id, name_iter in enumerate(parameter_name_ls):
    mcmc_summary_merge_dict[name_iter] = []
mcmc_summary_merge_dict['log_joint_prob'] = []
num_chain = 1
num_burnin = 3000
num_samples = 1000
iter_check_val = 100

if mcmc_bool:
    # import sim_data
    source_data, new_data = import_sim_data(sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length, True)

    # use the reference methods from training set with fewer sequence replications (no cheating)
    if seq_i > 0:
        mcmc_output_0_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
            ref_name_dir, 0, seq_i)
        )
        alpha_0_tar_ref_mean = np.mean(mcmc_output_0_dict['alpha_tar_0'], axis=0)

    else:
        eigen_val_tar = eigen_val_dict['target']
        alpha_0_tar_ref_mean = mvn(mean=np.zeros_like(eigen_val_tar), cov=np.diag(eigen_val_tar)).rvs(size=1)

    for chain_id in range(num_chain):
        print('\n chain id: {}\n'.format(chain_id + 1))
        estimate_rho_bool = True
        log_joint_prob_iter = 0

        mcmc_summary_dict = initialize_data_fast_calculation_sim(N_total, signal_length, seq_i, seq_source_size,
                                                                 new_data, eigen_fun_mat_dict, eigen_val_dict,
                                                                 scenario_name_dir)
        parameter_iter_ls = [mcmc_summary_dict[name_iter][0] for name_iter in parameter_name_ls]

        step_size_ls = [np.zeros([N_total]) + 0.1, np.zeros([N_total]) + 0.2, 0.1]
        accept_sum_ls = [np.zeros([N_total]), np.zeros([N_total]), np.zeros([N_total]), 0]
        # psi_tar_iter, sigma_iter, rho_iter, and psi_0_ntar_iter

        rho_fix_ls = []
        z_num_vec = np.zeros([N_total - 1])

        for iter_id in tqdm.tqdm(range(num_burnin+num_samples)):
            if iter_id % 100 == 0:
                print(iter_id)
                for name_id, name in enumerate(parameter_name_ls):
                    if name not in ['beta_tar', 'beta_0_ntar', 'alpha_tar', 'alpha_0_ntar']:
                        print('{}: {}'.format(name, parameter_iter_ls[name_id]))
                print('log_joint_prob: {}'.format(log_joint_prob_iter))

                print('z_vector has mean {}'.format(z_num_vec / iter_check_val))
                # reset z_vector summation
                z_num_vec = np.zeros([N_total - 1])

            parameter_iter_ls, accept_iter_ls, log_joint_prob_iter = update_gibbs_sampler_per_iteration(
                eigen_fun_mat_dict, eigen_val_dict, source_data, new_data, N_total, signal_length, source_name_ls,
                step_size_ls, estimate_rho_bool, seq_source_size, ref_name_dir, *parameter_iter_ls,
                prior_mean_alpha_0_tar=alpha_0_tar_ref_mean, psi_loc=psi_loc, psi_scale=psi_scale, rho_grid=rho_grid,
                sigma_loc=sigma_loc, sigma_scale=sigma_scale,
                approx_threshold=log_lhd_diff_approx)

            for n in range(N_total):
                for ni in range(2):
                    accept_sum_ls[ni][n] = accept_sum_ls[ni][n] + accept_iter_ls[ni][n]
            accept_sum_ls[2] = accept_sum_ls[2] + accept_iter_ls[2]
            z_num_vec = z_num_vec + parameter_iter_ls[-1]

            # fix rho and eta after certain MCMC samples, i.e., half of burn-in
            if iter_id < int(num_burnin * 0.75):
                rho_fix_ls.append(parameter_iter_ls[rho_name_id])

            if iter_id == int(num_burnin * 0.75):
                rho_fix = stats.mode(np.stack(rho_fix_ls, axis=0), axis=0)[0][0]
                print('fix!')
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

    # evaluate trace-plots for convergence check
    # rho and sigma per chain
    color_ls = ['red', 'blue', 'green']
    if prob_0_option_id == 1:
        rho_true_ls = [0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7]
        sigma_true_ls = [3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0]
    else:
        rho_true_ls = [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7]
        sigma_true_ls = [3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0]

    fig0, ax0 = plt.subplots(2, N_total, figsize=(20, 10))
    for n in range(N_total):
        for chain_id in range(num_chain):
            low_id = chain_id * num_samples
            upp_id = (chain_id + 1) * num_samples

            ax0[0, n].plot(mcmc_summary_merge_dict['rho'][low_id:upp_id, n],
                           label='rho, chain_{}'.format(chain_id + 1), color=color_ls[chain_id])
            ax0[0, n].legend(loc='upper right')
            ax0[0, n].axhline(y=rho_true_ls[n], linestyle='-')
            ax0[0, n].set_title('Participant {}'.format(n))
            ax0[0, n].set_ylim([0.0, 1.0])

            ax0[1, n].plot(mcmc_summary_merge_dict['sigma'][low_id:upp_id, n],
                           label='sigma, chain_{}'.format(chain_id + 1), color=color_ls[chain_id])
            ax0[1, n].legend(loc='upper right')
            ax0[1, n].axhline(y=sigma_true_ls[n], linestyle='-')
            ax0[1, n].set_title('Participant {}'.format(n))
            ax0[1, n].set_ylim([0, 10])

    fig0.savefig('{}/trace_plot_sub_0_seq_size_{}_{}_chains_cluster.png'.format(cluster_name_dir, seq_i + 1, num_chain))

else:
    # load mcmc_dict
    mcmc_summary_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster.mat'.format(cluster_name_dir, seq_i + 1)
    mcmc_summary_merge_dict = sio.loadmat(mcmc_summary_dict_dir)

    # mcmc_size_total = int(num_samples * num_chain)
    mcmc_size_total = num_chain * num_samples

    # mcmc_ids_seq_i = np.zeros([mcmc_size_per_chain, chain_size])
    # mcmc_sigma_0 = np.reshape(mcmc_summary_merge_dict['sigma'][:, 0], [mcmc_size_per_chain, chain_size])
    # mcmc_sigma_0_mean = np.mean(mcmc_sigma_0, axis=0)
    # for chain_id in range(chain_size):
    #     if np.abs(mcmc_sigma_0_mean[chain_id] - 3) <= 0.05:
    #         mcmc_ids_seq_i[:, chain_id] = 1
    # mcmc_ids_seq_i = np.reshape(mcmc_ids_seq_i, [mcmc_size_total])
    # if np.sum(mcmc_ids_seq_i) == 0:
    #     mcmc_ids_seq_i = mcmc_ids_seq_i + 1

    beta_tar_mcmc = mcmc_summary_merge_dict['beta_tar']
    beta_0_ntar_mcmc = mcmc_summary_merge_dict['beta_0_ntar']
    z_mcmc = mcmc_summary_merge_dict['z_vector']

    q_low = 0.05
    q_upp = 1 - q_low

    beta_tar_mean = [np.mean(beta_tar_mcmc[:, 0, :], axis=0)]
    beta_tar_median = [np.median(beta_tar_mcmc[:, 0, :], axis=0)]
    beta_tar_low = [np.quantile(beta_tar_mcmc[:, 0, :], axis=0, q=q_low)]
    beta_tar_upp = [np.quantile(beta_tar_mcmc[:, 0, :], axis=0, q=q_upp)]

    for nn in range(N_total-1):
        # beta_tar_mean.append(np.mean(beta_tar_mcmc[:, nn+1, :][z_mcmc[:, nn] == 0], axis=0))
        # beta_tar_low.append(np.quantile(beta_tar_mcmc[:, nn+1, :][z_mcmc[:, nn] == 0], axis=0, q=q_low))
        # beta_tar_upp.append(np.quantile(beta_tar_mcmc[:, nn+1, :][z_mcmc[:, nn] == 0], axis=0, q=q_upp))

        beta_tar_mean.append(np.mean(beta_tar_mcmc[:, nn + 1, :], axis=0))
        beta_tar_median.append(np.median(beta_tar_mcmc[:, nn + 1, :], axis=0))
        beta_tar_low.append(np.quantile(beta_tar_mcmc[:, nn + 1, :], axis=0, q=q_low))
        beta_tar_upp.append(np.quantile(beta_tar_mcmc[:, nn + 1, :], axis=0, q=q_upp))

    beta_tar_summary_dict = {
        'mean': np.stack(beta_tar_mean, axis=0),
        'median': np.stack(beta_tar_median, axis=0),
        'low': np.stack(beta_tar_low, axis=0),
        'upp': np.stack(beta_tar_upp, axis=0)
    }

    # beta_tar_summary_dict = {
    #     'mean': np.mean(beta_tar_mcmc, axis=0),
    #     # 'median': np.median(beta_tar_mcmc, axis=0),
    #     'low': np.quantile(beta_tar_mcmc, axis=0, q=q_low),
    #     'upp': np.quantile(beta_tar_mcmc, axis=0, q=q_upp)
    # }

    beta_0_ntar_summary_dict = {
        'mean': np.mean(beta_0_ntar_mcmc, axis=0),
        # 'median': np.median(beta_0_ntar_mcmc, axis=0),
        'low': np.quantile(beta_0_ntar_mcmc, axis=0, q=q_low),
        'upp': np.quantile(beta_0_ntar_mcmc, axis=0, q=q_upp)
    }

    fig1, ax1 = plt.subplots(2, 4, figsize=(18, 10))
    x_time = np.arange(signal_length)
    # true_group_k = [0, 0, 0, 1, 1, 2, 2]
    if prob_0_option_id == 1:
        subject_true_group_ls = [0, 1, 1, 1, 2, 2, 2]
    elif prob_0_option_id == 2:
        subject_true_group_ls = [0, 0, 0, 1, 1, 2, 2]
    else:
        subject_true_group_ls = []

    for n in range(N_total):
        true_group_k_n_name = 'group_{}'.format(subject_true_group_ls[n])
        n_row, n_col = int(n / 4), n % 4
        ax1[n_row, n_col].plot(x_time, rcp_function_smooth_dict_1[true_group_k_n_name]['target'], label='Target (true)',
                               color='red')
        ax1[n_row, n_col].plot(x_time, beta_tar_summary_dict['mean'][n, :], label='Target (mean)',
                               color='orange')
        ax1[n_row, n_col].fill_between(x_time, beta_tar_summary_dict['low'][n, :],
                                       beta_tar_summary_dict['upp'][n, :], alpha=0.2, label='Target',
                                       color='red')
        ax1[n_row, n_col].set_ylim([-4, 4])
        ax1[n_row, n_col].legend(loc='best')
        ax1[n_row, n_col].set_xlabel('Time (unit)')
        ax1[n_row, n_col].set_title('Subject {}'.format(n))
        if n == 0:
            n_row, n_col = int(7 / 4), 7 % 4
            ax1[n_row, n_col].plot(x_time, rcp_function_smooth_dict_1[true_group_k_n_name]['non-target'],
                                   label='Non-target (true)', color='blue')
            ax1[n_row, n_col].fill_between(x_time, beta_0_ntar_summary_dict['low'],
                                           beta_0_ntar_summary_dict['upp'], alpha=0.2, label='Non-target',
                                           color='blue')
            ax1[n_row, n_col].set_ylim([-4, 4])
            ax1[n_row, n_col].legend(loc='best')
            ax1[n_row, n_col].set_xlabel('Time (unit)')
            ax1[n_row, n_col].set_title('Subject {}'.format(n))
    fig1.savefig('{}/plot_sub_0_seq_size_{}_{}_chains_cluster.png'.format(cluster_name_dir, seq_i + 1, num_chain))
