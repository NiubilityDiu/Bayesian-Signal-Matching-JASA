from self_py_fun.SimGenFun import *
from self_py_fun.SimGlobal import *
import matplotlib.pyplot as plt
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_local)
    N = 7
    K = 3
    prob_0_option_id = 1
    sigma_val = 3.0
    rho_val = 0.6
    iter_total_num = 100
else:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_slurm)
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])
    iter_total_num = 100

prob_ls_enum = [
    # 2022-04-21 updates:
    # option id=1, N=7, K=3, no matching subjects
    [np.array([1.0, 0.0]),
     np.array([0.0, 1.0]), np.array([0.0, 1.0]),
     np.array([0.0, 1.0]), np.array([0.0, 0.0]),
     np.array([0.0, 0.0]), np.array([0.0, 0.0])],
    # option id=2, N=7, K=3, use data from group 0 to generate subject 0,
    # the remaining 6 are evenly split into three groups
    [np.array([1.0, 0.0]),
     np.array([1.0, 0.0]), np.array([1.0, 0.0]),
     np.array([0.0, 1.0]), np.array([0.0, 1.0]),
     np.array([0.0, 0.0]), np.array([0.0, 0.0])]
]
label_N_dict = generate_label_N_K_general(prob_ls_enum[prob_0_option_id-1], K)
print(label_N_dict)

# Scenario 1:
output_dir = '{}/numpyro_output'.format(parent_dir)
scenario_name_dir = '{}/N_{}_K_{}'.format(output_dir, N, K)

param_true_dict = {
    'target_char': target_char,
    'N': N,
    'seq_size': [seq_source_size for i in range(N)],
    'signal_length': signal_length,
    'label_N': label_N_dict,
    'group_0': {'beta_tar': rcp_function_smooth_dict_1['group_0']['target'],
                'beta_ntar': rcp_function_smooth_dict_1['group_0']['non-target'],
                'sigma': sigma_val,
                'rho': rho_val
                },
    'group_1': {'beta_tar': rcp_function_smooth_dict_1['group_1']['target'],
                'beta_ntar': rcp_function_smooth_dict_1['group_1']['non-target'],
                'sigma': sigma_val+1.0,
                'rho': rho_val
                },
    'group_2': {'beta_tar': rcp_function_smooth_dict_1['group_2']['target'],
                'beta_ntar': rcp_function_smooth_dict_1['group_2']['non-target'],
                'sigma': sigma_val,
                'rho': rho_val+0.1
                }
}

if K == 3:
    fig0, ax0 = plt.subplots(3, 1, figsize=(8, 18))
else:
    fig0, ax0 = plt.subplots(1, 4, figsize=(21, 6))

# for options 9-12, we use different orders to rank true ERP functions based on the magnitude values
original_K = np.arange(K)

for k, order_k in enumerate(original_K):
    print(k, order_k)
    group_k_name = 'group_{}'.format(k)
    ax0[order_k].plot(
        np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_tar'],
        label='target', color='red'
    )
    ax0[order_k].plot(
        np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_ntar'],
        label='non-target', color='blue'
    )
    ax0[order_k].legend(loc='lower right')
    ax0[order_k].set_xlabel('Time (unit)')
    ax0[order_k].set_ylim((-4, 4))
    ax0[k].set_title('True curve, {}'.format(group_k_name))
fig0.savefig('{}/N_{}_K_{}_option_{}_true_ERP_functions.png'.format(
    scenario_name_dir, N, K, prob_0_option_id)
)


target_char_test = list('THE0QUICK0BROWN0FOX')
seq_size_test = 10

if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

scenario_name = 'N_{}_K_{}_option_{}_sigma_{}_rho_{}'.format(
    N, K, prob_0_option_id, sigma_val, rho_val
)
scenario_name_dir = '{}/{}'.format(scenario_name_dir, scenario_name)
if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

for iter_num in range(iter_total_num):
    print('iteration {}'.format(iter_num))
    np.random.seed(iter_num)
    scenario_name_dir_try = '{}/iter_{}'.format(scenario_name_dir, iter_num)
    if not os.path.exists(scenario_name_dir_try):
        os.mkdir(scenario_name_dir_try)

    # single-channel data
    # generate the training and source data
    sim_data_dict = generate_simulated_data(param_true_dict, sim_rcp_array, scenario_name_dir_try, seed_num=iter_num)

    # examine the simulated training data
    fig1, ax1 = plt.subplots(3, 3, figsize=(16, 16))
    for i in range(N):
        subject_i_name = 'subject_{}'.format(i)
        sim_data_sub_i_X = np.reshape(
            np.array([sim_data_dict[subject_i_name]['X']]),
            [len(target_char) * seq_source_size * rcp_unit_flash_num, signal_length]
        )
        sim_data_sub_i_Y = np.reshape(
            np.array(sim_data_dict[subject_i_name]['Y']),
            [len(target_char) * seq_source_size * rcp_unit_flash_num]
        )
        mean_tar_i = np.mean(sim_data_sub_i_X[sim_data_sub_i_Y == 1, :], axis=0)
        mean_ntar_i = np.mean(sim_data_sub_i_X[sim_data_sub_i_Y != 1, :], axis=0)

        if N == 4:
            ax1_row = int(i / 2)
            ax1_col = i % 2
        else:
            ax1_row = int(i / 3)
            ax1_col = i % 3
        ax1[ax1_row, ax1_col].plot(np.arange(signal_length), mean_tar_i, label='target', color='red')
        ax1[ax1_row, ax1_col].plot(np.arange(signal_length), mean_ntar_i, label='non-target', color='blue')
        ax1[ax1_row, ax1_col].legend(loc='lower right')
        ax1[ax1_row, ax1_col].set_xlabel('Time (unit)')
        ax1[ax1_row, ax1_col].set_ylim([-5, 5])
        # ax1[0].set_ylim([-5, 6])
        ax1[ax1_row, ax1_col].set_title('{}, cluster_{}'.format(
            subject_i_name, sim_data_dict[subject_i_name]['label'])
        )
    fig1.savefig('{}/plot_sim_data_mean.png'.format(scenario_name_dir_try))

    # generate the testing data
    sim_data_test_dict = generate_simulated_test_data(param_true_dict, sim_data_dict['subject_0']['label'],
                                                      target_char_test, seq_size_test, sim_rcp_array,
                                                      scenario_name_dir_try, seed_num=iter_num)

