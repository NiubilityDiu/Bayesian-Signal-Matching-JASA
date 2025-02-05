from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.EEGGlobal import *
import tqdm
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
step_size_adjust_bool = True
N_total = len(sub_name_ls)
hyper_param_bool = True
xdawn_bool = True
run_bool = True
ref_mcmc_bool = False

select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)

if local_use:
    parent_dir = '{}/TRN_files'.format(parent_path_local)
    sub_new_name = 'K155'
    seq_source = 10
    seq_i = 5  # take values between 0 and 9.
    n_components = 2
    reduced_bool = True
    length_ls = [[0.30, 0.20]]
    gamma_val_ls = [[1.2, 1.2]]
    log_lhd_diff_approx = 1.0

else:
    parent_dir = '{}/TRN_files'.format(parent_path_slurm)
    # sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    sub_new_name = 'K151'
    seq_source = int(sys.argv[2])
    seq_i = int(sys.argv[3])
    n_components = int(sys.argv[4])
    reduced_bool = (sys.argv[5] == 'T' or sys.argv[5] == 'True')
    # length_ls = [[float(sys.argv[6]), float(sys.argv[7])]]
    # gamma_val_ls = [[float(sys.argv[8]), float(sys.argv[8])]]
    # sensitivity analysis check
    length_ls = length_super_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    gamma_val_ls = gamma_val_super_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    log_lhd_diff_approx = float(sys.argv[9])

upper_seq_id = 1 + seq_i
sub_new_dir = '{}/{}'.format(parent_dir, sub_new_name)
dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'

if reduced_bool:
    # Based on swLDA only (absolute testing accuracy over 50%), 23 or 24 subjects
    source_sub_name_ls = np.copy(sub_name_ls).tolist()
    sub_new_name_id = sub_name_ls.index(sub_new_name)
    del source_sub_name_ls[sub_new_name_id]

    for ele in sorted(sub_throw_name_ls, reverse=True):
        if ele in source_sub_name_ls:
            source_sub_name_ls.remove(ele)
    sub_name_reduce_ls = np.copy(np.array(source_sub_name_ls)).tolist()
    sub_name_reduce_ls.insert(0, sub_new_name)
    # sub_name_reduce_ls has either 24 or 25 elements,
    # the first element is always the new participant, while the remaining ones are source participants' names.

    # update N_total here if nothing is removed from the original 24-participant cohort.
    N_total = len(sub_name_reduce_ls)
    sub_new_cluster_dir = '{}/borrow_gibbs_letter_{}_reduced_xdawn'.format(sub_new_dir, letter_dim_sub)

else:
    # ignore this option.
    sub_name_reduce_ls = np.copy(sub_name_ls).tolist()
    sub_name_reduce_ls.remove(sub_new_name)
    sub_name_reduce_ls.insert(0, sub_new_name)
    source_sub_name_ls = np.copy(sub_name_ls).tolist()
    sub_new_cluster_dir = '{}/borrow_gibbs_letter_{}_xdawn'.format(sub_new_dir, letter_dim_sub)

xdawn_min = min(E_total, n_components)
channel_ids_sub = np.arange(xdawn_min)

if not os.path.exists('{}'.format(sub_new_cluster_dir)):
    os.mkdir(sub_new_cluster_dir)

sub_new_cluster_dir_2 = '{}/channel_all_comp_{}'.format(
    sub_new_cluster_dir, n_components
)
if not os.path.exists('{}'.format(sub_new_cluster_dir_2)):
    os.mkdir(sub_new_cluster_dir_2)

# sensitivity analysis check
kernel_name = 'length_{}_{}_gamma_{}'.format(length_ls[0][0], length_ls[0][1], gamma_val_ls[0][0])
print(kernel_name)
sub_new_cluster_dir_2 = '{}/{}'.format(sub_new_cluster_dir_2, kernel_name)
if not os.path.exists(sub_new_cluster_dir_2):
    os.mkdir(sub_new_cluster_dir_2)

# import eeg_data
source_data, new_data = import_eeg_data_cluster_format(
    sub_name_reduce_ls, sub_new_name, target_char_size, select_channel_ids, signal_length,
    letter_dim_sub, seq_i, seq_source, parent_dir, dat_name_common,
    xdawn_bool, reshape_2d_bool=False, n_components=n_components
)

# create the kernel matrix and related hyper-parameters
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls, gamma_val_ls, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']

# other hyper-parameters:
psi_loc, psi_scale = 0.0, 1.0
sigma_0_invgamma_loc = 3
sigma_0_invgamma_scale = 3
delta_mean = 1.0
delta_sd = 0.1
sigma_a = delta_mean**2/delta_sd**2+2
sigma_b = delta_mean * (delta_mean**2 / delta_sd**2 + 1)

param_name_ls = ['B_tar', 'B_0_ntar',
                 'A_tar', 'A_0_ntar',
                 'psi_tar', 'psi_0_ntar',
                 'sigma', 'rho', 'eta',
                 'z_vector']

rho_name_id = param_name_ls.index('rho')
eta_name_id = param_name_ls.index('eta')

mcmc_summary_chain_dict = {}

num_chain = 2
num_burnin = 8000
num_samples = 1000
estimate_rho_bool = False
estimate_eta_bool = False
ref_mcmc_id = None
iter_check_val = 100

if run_bool:
    for chain_id in range(num_chain):
        print('\n chain id: {}\n'.format(chain_id + 1))

        log_joint_prob_iter = 0
        mcmc_summary_dict = initialize_data_multi_fast_calculation_eeg(
            xdawn_min, xdawn_bool, letter_dim_sub,
            sub_name_reduce_ls, seq_source_size, eigen_fun_mat_dict, parent_dir,
            hyper_param_bool=hyper_param_bool, n_components=n_components,
            kernel_name=kernel_name
        )

        parameter_iter_ls = [np.copy(mcmc_summary_dict[name_iter][0]) for name_iter in param_name_ls]

        # mh_param_size = 3
        step_size_ls = [np.ones([N_total]) * 0.1, np.ones([N_total, xdawn_min]) * 0.01, 0.1]
        accept_sum_ls = [np.zeros(N_total), np.zeros([N_total, xdawn_min]), 0]
        # psi_0_tar, ..., psi_N_tar (N_total),
        # (sigma_0_1,...,sigma_0_E), ..., (sigma_N_1, ..., sigma_N_E) (N_total,E)
        # In fact, we do not estimate rho_0, ..., rho_N, eta_0, ..., eta_N,
        # nor (sigma_N_1, ..., sigma_n_E), n>0,
        # nor psi_0_ntar.
        # When we recycled all old parameters, we only cared about target parameters associated with the cluster 0.

        z_num_vec = np.zeros([N_total-1])

        for iter_id in tqdm.tqdm(range(num_burnin + num_samples)):

            parameter_iter_ls, accept_iter_ls, log_joint_prob_iter = update_multi_gibbs_sampler_per_iteration(
                eigen_fun_mat_dict, eigen_val_dict, source_data, new_data, N_total, xdawn_min, signal_length,
                sub_name_reduce_ls, step_size_ls, estimate_eta_bool, estimate_rho_bool,
                parent_dir, seq_source, seq_i+1, xdawn_bool, letter_dim_sub,
                *parameter_iter_ls,
                psi_loc=psi_loc, psi_scale=psi_scale, sub_new_name=sub_new_name,
                sigma_loc=sigma_0_invgamma_loc, sigma_scale=sigma_0_invgamma_scale,
                n_components=n_components,
                hyper_param_bool=hyper_param_bool, kernel_name=kernel_name,
                ref_mcmc_bool=ref_mcmc_bool, ref_mcmc_id=ref_mcmc_id, z_n_match_prior_prob=0.5,
                approx_threshold=log_lhd_diff_approx, approx_random_bool=True
            )

            for n in range(N_total):
                accept_sum_ls[0][n] = accept_sum_ls[0][n] + accept_iter_ls[0][n]
                accept_sum_ls[1][n, :] = accept_sum_ls[1][n, :] + accept_iter_ls[1][n, :]
            accept_sum_ls[2] = accept_sum_ls[2] + accept_iter_ls[2]

            z_num_vec = z_num_vec + parameter_iter_ls[-1]

            if iter_id % iter_check_val == 0:
                print('MCMC iteration number: {}'.format(iter_id))
                for name_id, name in enumerate(param_name_ls):
                    if name in ['B_0_ntar']:
                        print('{}: \n {}'.format(name, parameter_iter_ls[name_id].T))
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
                        for xdawn_e in range(xdawn_min):
                            step_size_ls[1][n, xdawn_e] = adjust_MH_step_size_random_walk(
                                step_size_ls[1][n, xdawn_e],
                                accept_mean_ls[1][n, xdawn_e]
                            )
                    step_size_ls[2] = adjust_MH_step_size_random_walk(step_size_ls[2], accept_mean_ls[2])

                    print('acceptance rate for psi_tar, sigma, and psi_0_ntar are \n{}, \n{}, and {}, respectively.'.format(
                        accept_mean_ls[0], accept_mean_ls[1].T, accept_mean_ls[2])
                    )
                    print('step_sizes for psi_tar, sigma, and psi_0_ntar are \n{}, \n{}, and {}, respectively.'.format(
                        step_size_ls[0], step_size_ls[1].T, step_size_ls[2])
                    )
                    # reset accept_sum_ls
                    accept_sum_ls = [np.zeros(N_total), np.zeros([N_total, xdawn_min]), 0]

            # record MCMC samples after num_burnin iterations.
            if iter_id >= num_burnin:
                # if iter_id > 0:
                for name_num_id, name_id in enumerate(param_name_ls):
                    mcmc_summary_dict[name_id].append(parameter_iter_ls[name_num_id])
                mcmc_summary_dict['log_joint_prob'].append(log_joint_prob_iter)

        for _, name_iter in enumerate(param_name_ls):
            print(name_iter)
            # remove the first initialization values
            if name_iter in ['psi_0_ntar']:
                mcmc_summary_dict[name_iter] = np.array(mcmc_summary_dict[name_iter], dtype='object')[1:]
            else:
                mcmc_summary_dict[name_iter] = np.stack(mcmc_summary_dict[name_iter], axis=0)[1:, ...]
        mcmc_summary_dict['log_joint_prob'] = mcmc_summary_dict['log_joint_prob'][1:]

        mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)] = {}
        mcmc_summary_chain_dict['chain_{}'.format(chain_id+1)] = mcmc_summary_dict

    ### save mcmc_dict
    mcmc_summary_chain_dict_dir = '{}/mcmc_seq_size_{}_cluster_xdawn_log_lhd_diff_approx_{}.mat'.format(
        sub_new_cluster_dir_2, seq_i + 1, log_lhd_diff_approx)
    sio.savemat(mcmc_summary_chain_dict_dir, mcmc_summary_chain_dict)



### load mcmc_dict
mcmc_summary_chain_dict_dir = '{}/mcmc_seq_size_{}_cluster_xdawn_log_lhd_diff_approx_{}.mat'.format(
    sub_new_cluster_dir_2, seq_i + 1, log_lhd_diff_approx)
mcmc_summary_chain_dict = sio.loadmat(mcmc_summary_chain_dict_dir, simplify_cells=True)

q_low = 0.05
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

every_5_sample_ids = np.arange(0, num_chain * num_samples, 2)
# every_5_sample_ids = np.arange(0, num_chain * num_samples)
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
if True:
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
x_time = np.arange(signal_length * xdawn_min)

produce_ERP_and_z_vector_plots(
    N_total, x_time,
    B_tar_summary_dict, B_0_ntar_summary_dict, sigma_mean, z_vector_mean,
    signal_length, xdawn_min, seq_i, sub_name_reduce_ls,
    sub_new_cluster_dir_2, num_chain, log_lhd_diff_approx=log_lhd_diff_approx,
    y_low=-2, y_upp=2
)


