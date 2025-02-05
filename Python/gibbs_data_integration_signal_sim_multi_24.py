from self_py_fun.MCMCMultiFun import *
import tqdm
plt.style.use("bmh")

step_size_adjust_bool = True
E = 2
mcmc_bool = True
plot_gibbs_bool = True
matrix_normal_bool = True
xdawn_bool, letter_dim_sub = False, target_char_size

parent_path_sim_dir  = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/SIM_files'.format(parent_path_sim_dir)

iter_num = 0
N_total = 24
K = 24
seq_i = 4

seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_multi_xdawn_eeg'.format(N_total, K)
scenario_name_dir = '{}/{}'.format(parent_data_dir, scenario_name)
scenario_iter_name_dir = '{}/iter_{}'.format(scenario_name_dir, iter_num)

sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_iter_name_dir, sim_data_name
)

cluster_name_dir = '{}/borrow_gibbs'.format(scenario_iter_name_dir)
if not os.path.exists(cluster_name_dir):
    os.mkdir(cluster_name_dir)

ref_name_dir = '{}/reference_numpyro'.format(scenario_iter_name_dir)
if not os.path.exists(ref_name_dir):
    os.mkdir(ref_name_dir)

length_ls_2 = [[0.3, 0.2]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']
eigen_fun_mat_tar = eigen_fun_mat_dict['target']

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

mcmc_summary_chain_dict = {}

num_chain = 2
num_burnin = 5000
num_samples = 1000
iter_check_val = 100

# import sim_data
source_data, new_data = import_sim_data(
    sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length, matrix_normal_bool
)

if mcmc_bool:
    estimate_rho_bool = False
    estimate_eta_bool = False

    for chain_id in range(num_chain):
        print('\n chain id: {}\n'.format(chain_id + 1))
        log_joint_prob_iter = 0

        # semi-supervised clustering problem
        mcmc_summary_dict = initialize_data_multi_fast_calculation_sim(N_total, E, signal_length, seq_i, seq_size_ls[0],
                                                                       new_data, eigen_fun_mat_dict, eigen_val_dict,
                                                                       scenario_iter_name_dir)
        parameter_iter_ls = [np.copy(mcmc_summary_dict[name_iter][0]) for name_iter in parameter_name_ls]
        # mh_param_size = 3
        step_size_ls = [np.ones([N_total]) * 0.1, np.ones([N_total, E]) * 0.1, 0.1]
        accept_sum_ls = [np.zeros(N_total), np.ones([N_total, E]), 0]

        eta_fix_ls = []
        rho_fix_ls = []
        z_num_vec = np.zeros([N_total-1])

        for iter_id in tqdm.tqdm(range(num_burnin+num_samples)):

            parameter_iter_ls, accept_iter_ls, log_joint_prob_iter = update_multi_gibbs_sampler_per_iteration(
                eigen_fun_mat_dict, eigen_val_dict, source_data, new_data, N_total, E, signal_length, None,
                step_size_ls, estimate_eta_bool, estimate_rho_bool,
                ref_name_dir, seq_source_size, seq_i, xdawn_bool, letter_dim_sub,
                *parameter_iter_ls,
                psi_loc=psi_loc, psi_scale=psi_scale, sub_new_name=0,
                sigma_loc=sigma_loc, sigma_scale=sigma_scale,
                eta_grid=eta_grid, rho_grid=rho_grid, ref_mcmc_bool=False, ref_mcmc_id=None,
                approx_threshold=1.0, z_n_match_prior_prob=0.5, approx_random_bool=True
            )

            for n in range(N_total):
                accept_sum_ls[0][n] = accept_sum_ls[0][n] + accept_iter_ls[0][n]
                accept_sum_ls[1][n, :] = accept_sum_ls[1][n, :] + accept_iter_ls[1][n, :]
            accept_sum_ls[2] = accept_sum_ls[2] + accept_iter_ls[2]

            z_num_vec = z_num_vec + parameter_iter_ls[-1]

            if iter_id % iter_check_val == 0:
                print('MCMC iteration number: {}'.format(iter_id))
                for name_id, name in enumerate(parameter_name_ls):
                    if name not in ['B_tar', 'B_0_ntar', 'A_tar', 'A_0_ntar']:
                        print('{}: \n {}'.format(name, parameter_iter_ls[name_id]))
                print('log_joint_prob: \n {} \n'.format(log_joint_prob_iter))

                print('z_vector has mean {}'.format(z_num_vec / iter_check_val))
                # reset z_vector summation
                z_num_vec = np.zeros([N_total-1])

                # check step size and acceptance rate
                if step_size_adjust_bool and 0 < iter_id < int(num_burnin * 0.6):
                    accept_mean_ls = [accept_sum_ls[0] / iter_check_val,
                                      accept_sum_ls[1] / iter_check_val,
                                      accept_sum_ls[2] / iter_check_val]
                    for n in range(N_total):
                        step_size_ls[0][n] = adjust_MH_step_size_random_walk(step_size_ls[0][n], accept_mean_ls[0][n])
                        for e_iter in range(E):
                            step_size_ls[1][n, e_iter] = adjust_MH_step_size_random_walk(
                                step_size_ls[1][n, e_iter],
                                accept_mean_ls[1][n, e_iter]
                            )
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

            if estimate_eta_bool and iter_id < int(num_burnin * 0.8):
                eta_fix_ls.append(parameter_iter_ls[eta_name_id])
            if estimate_rho_bool and iter_id < int(num_burnin * 0.8):
                rho_fix_ls.append(parameter_iter_ls[rho_name_id])

            if estimate_eta_bool and iter_id == int(num_burnin * 0.8):
                eta_fix = stats.mode(np.stack(eta_fix_ls, axis=0), axis=0)[0][0]
                print('eta fix!')
                parameter_iter_ls[eta_name_id] = np.copy(eta_fix)
                estimate_eta_bool = False

            if estimate_rho_bool and iter_id == int(num_burnin * 0.8):
                rho_fix = stats.mode(np.stack(rho_fix_ls, axis=0), axis=0)[0][0]
                print('rho fix!')
                parameter_iter_ls[rho_name_id] = np.copy(rho_fix)
                estimate_rho_bool = False

            # record MCMC samples after num_burnin iterations.
            if iter_id >= num_burnin:
                for name_id, name in enumerate(parameter_name_ls):
                    mcmc_summary_dict[name].append(np.copy(parameter_iter_ls[name_id]))
                mcmc_summary_dict['log_joint_prob'].append(log_joint_prob_iter)

        for name_id, name_iter in enumerate(parameter_name_ls):
            print(name_iter)
            # remove the first initialization values
            if name_iter in ['psi_0_ntar']:
                mcmc_summary_dict[name_iter] = np.array(mcmc_summary_dict[name_iter], dtype='object')[1:]
            else:
                mcmc_summary_dict[name_iter] = np.stack(mcmc_summary_dict[name_iter], axis=0)[1:, ...]
        mcmc_summary_dict['log_joint_prob'] = mcmc_summary_dict['log_joint_prob'][1:]

        mcmc_summary_chain_dict['chain_{}'.format(chain_id + 1)] = {}
        mcmc_summary_chain_dict['chain_{}'.format(chain_id + 1)] = mcmc_summary_dict

    # save mcmc_dict
    mcmc_summary_chain_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster_log_lhd_diff_approx_{}.mat'.format(
        cluster_name_dir, seq_i + 1, log_lhd_diff_approx)
    sio.savemat(mcmc_summary_chain_dict_dir, mcmc_summary_chain_dict)

### load mcmc_dict
mcmc_summary_chain_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster_log_lhd_diff_approx_{}.mat'.format(
    cluster_name_dir, seq_i + 1, log_lhd_diff_approx)
mcmc_summary_chain_dict = sio.loadmat(mcmc_summary_chain_dict_dir, simplify_cells=True)

q_low = 0.025
q_upp = 1 - q_low
B_tar_mcmc = []
B_0_ntar_mcmc = []
sigma_mcmc = []
z_vec_mcmc = []

for chain_id in range(num_chain):
    B_tar_mcmc.append(mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)]['B_tar'])
    B_0_ntar_mcmc.append(mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)]['B_0_ntar'])
    sigma_mcmc.append(mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)]['sigma'])
    z_vec_mcmc.append(mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)]['z_vector'])

# every_5_sample_ids = np.arange(0, num_chain * num_samples, 5)
every_5_sample_ids = np.arange(0, num_chain * num_samples)
B_tar_mcmc = np.concatenate(B_tar_mcmc, axis=0)[every_5_sample_ids, ...]
B_0_ntar_mcmc = np.concatenate(B_0_ntar_mcmc, axis=0)[every_5_sample_ids, ...]
sigma_mcmc = np.concatenate(sigma_mcmc, axis=0)[every_5_sample_ids, ...]
z_vec_mcmc = np.concatenate(z_vec_mcmc, axis=0)[every_5_sample_ids, ...]

sigma_mean = np.round(np.mean(sigma_mcmc, axis=0), decimals=2)
z_vector_mean = np.mean(z_vec_mcmc, axis=0)

B_tar_summary_dict = {
    'mean': np.mean(B_tar_mcmc, axis=0),
    'low': np.quantile(B_tar_mcmc, axis=0, q=q_low),
    'upp': np.quantile(B_tar_mcmc, axis=0, q=q_upp)
}

# adjust new participant by only using the MCMC samples
# of which z_vec_mcmc contains non-zero value(s).
if False:
    B_0_tar_mcmc = B_tar_mcmc[:, 0, ...]
    B_0_tar_mcmc = B_0_tar_mcmc[np.sum(z_vec_mcmc, axis=1)>0, ...]
    B_tar_summary_dict['mean'][0, ...] = np.copy(np.mean(B_0_tar_mcmc, axis=0))
    B_tar_summary_dict['low'][0, ...] = np.copy(np.quantile(B_0_tar_mcmc, axis=0, q=q_low))
    B_tar_summary_dict['upp'][0, ...] = np.copy(np.quantile(B_0_tar_mcmc, axis=0, q=q_upp))


B_0_ntar_summary_dict = {
    'mean': np.mean(B_0_ntar_mcmc, axis=0),
    'low': np.quantile(B_0_ntar_mcmc, axis=0, q=q_low),
    'upp': np.quantile(B_0_ntar_mcmc, axis=0, q=q_upp)
}
x_time = np.arange(signal_length * E)

produce_ERP_and_z_vector_plots(
    N_total, x_time,
    B_tar_summary_dict, B_0_ntar_summary_dict, sigma_mean, z_vector_mean,
    signal_length, E, seq_i, np.arange(N_total).tolist(),
    cluster_name_dir, num_chain, log_lhd_diff_approx=log_lhd_diff_approx,
    y_low=-2, y_upp=2
)



