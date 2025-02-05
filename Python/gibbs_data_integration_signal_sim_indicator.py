from self_py_fun.MCMCFun import *
import tqdm
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
target_char = 'TT'
target_char_size = len(target_char)
# step_size_adjust_bool = False
E = 1


if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    sim_erp_multi_dir = '{}/SIM_files/Chapter_3/sim_erp_multi.mat'.format(parent_path_local)
    # iter_num = 1
    N_total = 7
    K = 3
    prob_0_option_id = 2
    sigma_val = 3.0
    rho_val = 0.6

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    sim_erp_multi_dir = '{}/SIM_files/Chapter_3/sim_erp_multi.mat'.format(parent_path_slurm)
    # iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N_total = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])

# np.random.seed(iter_num)
seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_option_{}_sigma_{}_rho_{}'.format(N_total, K, prob_0_option_id, sigma_val, rho_val)
scenario_name_dir = '{}/N_{}_K_{}/{}'.format(parent_dir, N_total, K, scenario_name)

length_ls_2 = [[0.4, 0.3]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']
eigen_fun_mat_tar = eigen_fun_mat_dict['target']
approx_threshold = 5.0
eta_grid = np.round(np.arange(0, 1, 0.05), 2).tolist()
rho_grid = np.round(np.arange(0, 1, 0.05), 2).tolist()
num_chain = 3
num_samples = 300
borrow_mcmc_size = num_samples * num_chain
sim_replication_size = 100

data_log_prob_vec_mat = []
z_vec_mat = []


for iter_num in range(sim_replication_size):
    print('iter_{}'.format(iter_num))

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

    # load mcmc_dict
    for seq_i in range(seq_source_size):
        mcmc_summary_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster.mat'.format(cluster_name_dir, seq_i + 1)
        mcmc_summary_merge_dict = sio.loadmat(mcmc_summary_dict_dir)
        color_ls = ['red', 'blue', 'green']

        # testing section
        source_data, new_data = import_sim_data(sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length)

        mcmc_output_borrow_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_cluster.mat'.format(
            cluster_name_dir, 0, seq_i + 1
        ))

        for borrow_mcmc_id in np.arange(0, borrow_mcmc_size, 5):
            z_vector_iter_2 = np.zeros([N_total - 1])
            # z_vector_prob_iter_2 = np.zeros([N_total - 1])

            beta_0_tar_iter = mcmc_output_borrow_dict['beta_tar'][borrow_mcmc_id, 0, :]
            sigma_0_iter = mcmc_output_borrow_dict['sigma'][borrow_mcmc_id, 0]
            rho_0_iter = mcmc_output_borrow_dict['rho'][borrow_mcmc_id, 0]
            # eta_0_iter = mcmc_output_borrow_dict['eta'][borrow_mcmc_id, 0]

            for n in range(N_total-1):
                name_n = 'subject_{}'.format(n + 1)
                source_data_n = source_data[name_n]
                # import mcmc from reference methods:
                # mcmc_output_ref_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                #     ref_name_dir, n + 1, seq_source_size)
                # )
                #
                # alpha_n_tar_ref_iter = np.mean(mcmc_output_ref_dict['alpha_tar_0'], axis=0)
                # sigma_n_ref_iter = np.mean(mcmc_output_ref_dict['sigma_0'])
                # psi_n_tar_ref_iter = np.mean(mcmc_output_ref_dict['psi_tar_0'])
                # beta_n_tar_ref_iter = psi_n_tar_ref_iter * eigen_fun_mat_tar @ alpha_n_tar_ref_iter
                #
                # rho_n_ref_iter = find_nearby_parameter_grid(np.mean(mcmc_output_ref_dict['rho_0']), rho_grid)

                beta_n_tar_ref_iter = mcmc_output_borrow_dict['beta_tar'][borrow_mcmc_id, n+1, :]
                sigma_n_ref_iter = mcmc_output_borrow_dict['sigma'][borrow_mcmc_id, n+1]
                rho_n_ref_iter = mcmc_output_borrow_dict['rho'][borrow_mcmc_id, n+1]

                data_log_prob_vec_iter, _, z_vector_iter_2_n = update_indicator_n(
                    beta_0_tar_iter, beta_n_tar_ref_iter, sigma_0_iter, sigma_n_ref_iter, rho_0_iter, rho_n_ref_iter,
                    source_data_n, signal_length
                    # approx_threshold=approx_threshold
                )
                z_vector_iter_2[n] = z_vector_iter_2_n
                # z_vector_prob_iter_2[n] = z_vector_prob_iter_2_n
                data_log_prob_vec_mat.append(data_log_prob_vec_iter)

                # beta_tar_iter = mcmc_output_borrow_dict['beta_tar'][borrow_mcmc_id, ...]
                # beta_0_ntar_iter = mcmc_output_borrow_dict['beta_0_ntar'][borrow_mcmc_id, :]
                # sigma_iter = mcmc_output_borrow_dict['sigma'][borrow_mcmc_id, :]
                # rho_iter = mcmc_output_borrow_dict['rho'][borrow_mcmc_id, :]
                # z_vector_iter = np.array([0, 0, 0, 0, 0, 0])
                # signal_length = 35
                # seq_source_size = 10
                #
                #
                # sigma_iter, sigma_n_accept = update_sigma_n_RWMH(beta_tar_iter, beta_0_ntar_iter, sigma_iter, rho_iter,
                #                                                  z_vector_iter, source_data, new_data, 7, n,
                #                                                  signal_length,
                #                                                  seq_source_size, ref_name_dir, 100,
                #                                                  0.5, 0.0, 5.0, None)

            z_vec_mat.append(np.copy(z_vector_iter_2))


data_log_prob_vec_mat = np.reshape(
    np.stack(data_log_prob_vec_mat, axis=0), [sim_replication_size, seq_source_size, int(borrow_mcmc_size/5), N_total - 1, 2]
)
z_vec_mat = np.reshape(
    np.stack(z_vec_mat, axis=0), [sim_replication_size, seq_source_size, int(borrow_mcmc_size/5), N_total - 1]
)

data_log_prob_vec_diff_mat = data_log_prob_vec_mat[..., 1] - data_log_prob_vec_mat[..., 0]  # compared to subject 0
data_log_prob_vec_diff_mean_mat = np.mean(data_log_prob_vec_diff_mat, axis=(0, 2))
data_log_prob_vec_diff_se_mat = np.sqrt(np.var(data_log_prob_vec_diff_mat, axis=(0, 2)))

z_vec_mat_mean = np.mean(z_vec_mat, axis=(0, 2))
z_vec_mat_se = np.sqrt(np.var(z_vec_mat, axis=(0, 2)))

# save as .mat file
inference_summary_dir = '{}/inference_summary'.format(scenario_name_dir)
if not os.path.exists('{}'.format(inference_summary_dir)):
    os.mkdir(inference_summary_dir)
z_vec_file_dir = '{}/z_indicator_summary.mat'.format(inference_summary_dir)

sio.savemat(
    z_vec_file_dir,
    {
        'z_mean': z_vec_mat_mean,
        'z_se': z_vec_mat_se,
        'log_prob_diff_mean': data_log_prob_vec_diff_mean_mat,
        'log_prob_diff_se': data_log_prob_vec_diff_se_mat
    }
)
