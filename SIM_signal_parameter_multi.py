import sys
import matplotlib.pyplot as plt
from self_py_fun.SimGenFun import *
plt.style.use("bmh")


local_use = True
# local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
target_char = 'T'
signal_length = 25
seq_size = 50

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_local)
    N = 4
    K = 3
    prob_0_option_id = 1
    rho_val = 0.5  # temporal
    lambda_val = 0.3  # spatial
    iter_total_num = 10
else:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_slurm)
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    rho_val = float(sys.argv[5])
    lambda_val = float(sys.argv[6])
    iter_total_num = 100

prob_ls_enum = [
    # option id=0: no match:
    [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])],
    # option id=1, evenly split for the remaining three:
    [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
]
label_N_dict = generate_label_N_K_general(prob_ls_enum[prob_0_option_id], K)
print(label_N_dict)


# Scenario 2:
# Try multi-channel data generation
output_dir = '{}/numpyro_output'.format(parent_dir)
source_dir = '{}/eeg_signal_source'.format(parent_dir)

'''
source_K114_dir = '{}/EEGDataTrunc/K114_001_BCI_TRN_Truncated_Data.mat'.format(source_dir)
source_K123_dir = '{}/EEGDataTrunc/K123_001_BCI_TRN_Truncated_Data.mat'.format(source_dir)
source_K151_dir = '{}/EEGDataTrunc/K151_001_BCI_TRN_Truncated_Data.mat'.format(source_dir)
source_K183_dir = '{}/EEGDataTrunc/K183_001_BCI_TRN_Truncated_Data.mat'.format(source_dir)
data_grp_0 = sio.loadmat(source_K114_dir)
data_grp_1 = sio.loadmat(source_K123_dir)
data_grp_2 = sio.loadmat(source_K151_dir)
data_grp_3 = sio.loadmat(source_K183_dir)

signal_grp_0 = data_grp_0['Signal']
signal_grp_1 = data_grp_1['Signal']
signal_grp_2 = data_grp_2['Signal']
signal_grp_3 = data_grp_3['Signal']

type_grp_0 = np.squeeze(data_grp_0['Type'], axis=-1)
type_grp_1 = np.squeeze(data_grp_1['Type'], axis=-1)
type_grp_2 = np.squeeze(data_grp_2['Type'], axis=-1)
type_grp_3 = np.squeeze(data_grp_3['Type'], axis=-1)

tar_grp_0 = np.reshape(np.mean(signal_grp_0[type_grp_0 == 1, :], axis=0), [16, 25])
tar_grp_1 = np.reshape(np.mean(signal_grp_1[type_grp_1 == 1, :], axis=0), [16, 25])
tar_grp_2 = np.reshape(np.mean(signal_grp_2[type_grp_2 == 1, :], axis=0), [16, 25])
tar_grp_3 = np.reshape(np.mean(signal_grp_3[type_grp_3 == 1, :], axis=0), [16, 25])

ntar_grp_0 = np.reshape(np.mean(signal_grp_0[type_grp_0 != 1, :], axis=0), [16, 25])
ntar_grp_1 = np.reshape(np.mean(signal_grp_1[type_grp_1 != 1, :], axis=0), [16, 25])
ntar_grp_2 = np.reshape(np.mean(signal_grp_2[type_grp_2 != 1, :], axis=0), [16, 25])
ntar_grp_3 = np.reshape(np.mean(signal_grp_3[type_grp_3 != 1, :], axis=0), [16, 25])
'''

source_K114_dir = '{}/MCMC/K114_mcmc_source.mat'.format(source_dir)
source_K123_dir = '{}/MCMC/K123_mcmc_source.mat'.format(source_dir)
source_K151_dir = '{}/MCMC/K151_mcmc_source.mat'.format(source_dir)
source_K183_dir = '{}/MCMC/K183_mcmc_source.mat'.format(source_dir)
mcmc_grp_0 = sio.loadmat(source_K114_dir)
mcmc_grp_1 = sio.loadmat(source_K123_dir)
mcmc_grp_2 = sio.loadmat(source_K151_dir)
mcmc_grp_3 = sio.loadmat(source_K183_dir)

tar_grp_0 = np.mean(mcmc_grp_0['beta_tar'], axis=(0, 3)) * 1.5
tar_grp_1 = np.mean(mcmc_grp_1['beta_tar'], axis=(0, 3)) * 1.5
tar_grp_2 = np.mean(mcmc_grp_2['beta_tar'], axis=(0, 3)) * 1.5
tar_grp_3 = np.mean(mcmc_grp_3['beta_tar'], axis=(0, 3)) * 1.5

ntar_grp_0 = np.mean(mcmc_grp_0['beta_ntar'], axis=(0, 3))
ntar_grp_1 = np.mean(mcmc_grp_1['beta_ntar'], axis=(0, 3))
ntar_grp_2 = np.mean(mcmc_grp_2['beta_ntar'], axis=(0, 3))
ntar_grp_3 = np.mean(mcmc_grp_3['beta_ntar'], axis=(0, 3))

var_K114_dir = '{}/Misc/K114_marginal_variance.mat'.format(source_dir)
var_K123_dir = '{}/Misc/K123_marginal_variance.mat'.format(source_dir)
var_K151_dir = '{}/Misc/K151_marginal_variance.mat'.format(source_dir)
var_K183_dir = '{}/Misc/K183_marginal_variance.mat'.format(source_dir)

var_grp_0 = np.floor(sio.loadmat(var_K114_dir)['variance'])
var_grp_1 = np.floor(sio.loadmat(var_K123_dir)['variance'])
var_grp_2 = np.floor(sio.loadmat(var_K151_dir)['variance'])
var_grp_3 = np.floor(sio.loadmat(var_K183_dir)['variance'])

# channel selection
# Cz, P3, Pz, P4, PO7, PO8, Oz
# channel_sub_ids = np.arange(16)
# channel_sub_ids = np.array([6, 11, 12, 13, 14, 15, 16]) - 1
# channel_sub_ids = np.array([6, 12, 13, 14, 15, 16]) - 1
channel_sub_ids = np.array([6, 12]) - 1

param_true_dict = {
    'target_char': target_char,
    'N': N,
    'seq_size': [seq_size for i in range(N)],
    'signal_length': signal_length,
    'label_N': label_N_dict,
    'channel_dim': len(channel_sub_ids),
    'group_0': {'beta_tar': tar_grp_0[channel_sub_ids, :],
                'beta_ntar': ntar_grp_0[channel_sub_ids, :],
                'sigma_sq': var_grp_0[channel_sub_ids],
                'rho': rho_val,
                'lambda': lambda_val
                },
    'group_1': {'beta_tar': tar_grp_1[channel_sub_ids, :],
                'beta_ntar': ntar_grp_1[channel_sub_ids, :],
                'sigma_sq': var_grp_1[channel_sub_ids],
                'rho': rho_val,
                'lambda': lambda_val
                },
    'group_2': {'beta_tar': tar_grp_2[channel_sub_ids, :],
                'beta_ntar': ntar_grp_2[channel_sub_ids, :],
                'sigma_sq': var_grp_2[channel_sub_ids],
                'rho': rho_val,
                'lambda': lambda_val
                },
    'group_3': {'beta_tar': tar_grp_3[channel_sub_ids, :],
                'beta_ntar': ntar_grp_3[channel_sub_ids, :],
                'sigma_sq': var_grp_3[channel_sub_ids],
                'rho': rho_val,
                'lambda': lambda_val
                }
}

target_char_test = list('THE0QUICK0BROWN0FOX')
seq_size_test = 10

scenario_name_dir = '{}/N_{}_K_{}'.format(output_dir, N, K)

if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

if K == 3:
    scenario_name = 'N_{}_K_{}_K114_K123_K151_based_option_{}_rho_{}_lambda_{}'.format(
        N, K, prob_0_option_id, rho_val, lambda_val
    )
else:
    scenario_name = 'N_{}_K_{}_K114_K123_K151_K183_based_option_{}_rho_{}_lambda_{}'.format(
        N, K, prob_0_option_id, rho_val, lambda_val
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

    # Scenario 2:
    # Multi-channel simulation data
    sim_data_dict = generate_simulated_data_multi_channel(
        param_true_dict, sim_rcp_array, scenario_name_dir_try, seed_num=iter_num
    )

    # examine the simulated training data
    for i in range(N):
        subject_i_name = 'subject_{}'.format(i)
        sim_data_dict[subject_i_name]['X'] = np.squeeze(np.array([sim_data_dict[subject_i_name]['X']]), axis=0)
        sim_data_dict[subject_i_name]['Y'] = np.array(sim_data_dict[subject_i_name]['Y'])
        mean_tar_i = np.mean(sim_data_dict[subject_i_name]['X'][sim_data_dict[subject_i_name]['Y'] == 1.0, ...], axis=0)
        mean_ntar_i = np.mean(sim_data_dict[subject_i_name]['X'][sim_data_dict[subject_i_name]['Y'] != 1.0, ...], axis=0)
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 8))
        for channel_e in range(len(channel_sub_ids)):
            # ax1_row = int(channel_e / 1)
            # ax1_col = channel_e % 1
            # channel_i = ax1_row * 3 + ax1_col
            ax1[channel_e].plot(
                np.arange(param_true_dict['signal_length']), mean_tar_i[channel_e, :], label='target', color='red'
            )
            ax1[channel_e].plot(
                np.arange(param_true_dict['signal_length']), mean_ntar_i[channel_e, :], label='non-target', color='blue'
            )
            ax1[channel_e].legend(loc='lower right')
            ax1[channel_e].set_xlabel('Time (unit)')
            ax1[channel_e].set_title('{}, cluster_{}, channel_{}'.format(
                subject_i_name, sim_data_dict[subject_i_name]['label'], channel_e+1)
            )
            fig1.savefig('{}/plot_sim_data_mean_{}.png'.format(scenario_name_dir_try, subject_i_name))

    # generate the testing data
    sim_data_test_dict = generate_simulated_test_data_multi_channel(
        param_true_dict, sim_data_dict['subject_0']['label'],
        target_char_test, seq_size_test, sim_rcp_array,
        scenario_name_dir_try, seed_num=iter_num
    )

