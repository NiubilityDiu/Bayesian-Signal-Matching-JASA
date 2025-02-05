from self_py_fun.SimGenFun import *
import matplotlib.pyplot as plt
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
E = 2
plot_sim_data_bool = True

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_local)
    N = 7
    K = 3
    prob_0_option_id = 2
    iter_total_num = 1
else:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_slurm)
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id =  int(sys.argv[4])
    iter_total_num =  int(sys.argv[5])

prob_ls_enum = [
    # option id=1: no match:
    [np.array([1.0, 0.0]),
     np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]),
     np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
    # option id=2, two matches:
    [np.array([1.0, 0.0]),
     np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
     np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
]
label_N_dict = generate_label_N_K_general(prob_ls_enum[prob_0_option_id - 1], K)

sigma_ls = [np.array([8.0, 8.0]), np.array([8.0, 6.0]), np.array([2.0, 2.0])]
rho_ls = [0.7, 0.7, 0.5]
eta_ls = [0.6, 0.4, 0.4]

# create rcp_function_multi_smooth_dict_1
sim_erp_multi_dir = '{}/sim_erp_multi.mat'.format(parent_dir)
sim_erp_multi = sio.loadmat(sim_erp_multi_dir)
rcp_function_multi_smooth_dict_1 = {
    'group_0': {'target': sim_erp_multi['group_0'][..., 0].T, 'non-target': sim_erp_multi['group_0'][..., 1].T},
    'group_1': {'target': sim_erp_multi['group_1'][..., 0].T, 'non-target': sim_erp_multi['group_1'][..., 1].T},
    'group_2': {'target': sim_erp_multi['group_2'][..., 0].T, 'non-target': sim_erp_multi['group_2'][..., 1].T}
}

output_dir = '{}/numpyro_output'.format(parent_dir)
param_true_dict = {
    'target_char': target_char,
    'N': N,
    'E': E,
    'seq_size': [seq_source_size for i in range(N)],
    'signal_length': signal_length,
    'label_N': label_N_dict,
    'group_0': {'beta_tar': rcp_function_multi_smooth_dict_1['group_0']['target'],
                'beta_ntar': rcp_function_multi_smooth_dict_1['group_0']['non-target'],
                'sigma': sigma_ls[0],
                'rho': rho_ls[0],
                'eta': eta_ls[0]
                },
    'group_1': {'beta_tar': rcp_function_multi_smooth_dict_1['group_1']['target'],
                'beta_ntar': rcp_function_multi_smooth_dict_1['group_1']['non-target'],
                'sigma': sigma_ls[1],
                'rho': rho_ls[1],
                'eta': eta_ls[1]
                },
    'group_2': {'beta_tar': rcp_function_multi_smooth_dict_1['group_2']['target'],
                'beta_ntar': rcp_function_multi_smooth_dict_1['group_2']['non-target'],
                'sigma': sigma_ls[2],
                'rho': rho_ls[2],
                'eta': eta_ls[2]
                }
}

target_char_test = list('THE0QUICK0BROWN0FOX')
seq_size_test = 10

scenario_name_dir = '{}/N_{}_K_{}'.format(output_dir, N, K)

if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

fig0, ax0 = plt.subplots(E, K, figsize=(18, 12))
for k in range(K):
    group_k_name = 'group_{}'.format(k)
    for e in range(E):
        ax0[e, k].plot(
            np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_tar'][e, :],
            label='target', color='red'
        )
        ax0[e, k].plot(
            np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_ntar'][e, :],
            label='non-target', color='blue'
        )
        ax0[e, k].legend(loc='lower right')
        ax0[e, k].set_xlabel('Time (unit)')
        ax0[e, k].set_ylim((-5, 5))
        ax0[e, k].set_title('{}, channel_{}'.format(group_k_name, e+1))
fig0.savefig('{}/N_{}_K_{}_multi_option_{}_true_ERP_functions.png'.format(
    scenario_name_dir, N, K, prob_0_option_id)
)

scenario_name = 'N_{}_K_{}_multi_option_{}'.format(N, K, prob_0_option_id)
scenario_name_dir = '{}/{}'.format(scenario_name_dir, scenario_name)
if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)


for iter_num in range(iter_total_num):
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
                ax1[e].set_ylim([-4, 4])
                ax1[e].set_title('{}, cluster_{}, channel_{}'.format(
                    subject_i_name, sim_data_dict[subject_i_name]['label'], e + 1)
                )
                fig1.savefig('{}/plot_sim_data_mean_{}.png'.format(scenario_name_dir_try, subject_i_name))

    # generate the testing data
    sim_data_test_dict = generate_simulated_test_data_multi(
        param_true_dict, sim_data_dict['subject_0']['label'], target_char_test,
        seq_size_test, scenario_name_dir_try, seed_num=iter_num
    )


