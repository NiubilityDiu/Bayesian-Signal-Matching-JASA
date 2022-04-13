import sys
import matplotlib.pyplot as plt
from self_py_fun.SimGenFun import *
plt.style.use("bmh")

local_use = True
# local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
target_char = 'T'
seq_size = 10

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_local)
    N = 9
    K = 4
    prob_0_option_id = 9
    sigma_val = 5.0
    rho_val = 0.5
    iter_total_num = 10
else:
    parent_dir = '{}/SIM_files/Chapter_3'.format(parent_path_slurm)
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])
    iter_total_num = 100

if K == 2:
    # for K=2 only (4 options):
    prob_0_ls_enum = [
        [1.0, 0.1, 0.1, 0.1],
        [1.0, 0.9, 0.1, 0.1],
        [1.0, 0.9, 0.9, 0.1],
        [1.0, 0.9, 0.9, 0.9]
    ]
    label_N_dict = generate_label_N_K_general(prob_0_ls_enum[prob_0_option_id], K)

elif K == 3:
    # for K=3 only (currently 8 options):
    prob_ls_enum = [
        # option id=0: no match, source 1-2 from cluster 1, source 3 from cluster 2
        [np.array([1.0, 0.0]), np.array([0.05, 0.9]), np.array([0.05, 0.9]), np.array([0.05, 0.05])],
        # option id=1: no match, source 1 from cluster 1, source 2-3 from cluster 2
        [np.array([1.0, 0.0]), np.array([0.05, 0.9]), np.array([0.05, 0.05]), np.array([0.05, 0.05])],
        # option id=2: one match, source 1 from cluster 0, source 2 from cluster 1, source 3 from cluster 2
        [np.array([1.0, 0.0]), np.array([0.9, 0.05]), np.array([0.05, 0.9]), np.array([0.05, 0.05])],
        # option id=3: two matches, source 1-2 from cluster 0, source 3 from cluster 1
        [np.array([1.0, 0.0]), np.array([0.9, 0.05]), np.array([0.9, 0.05]), np.array([0.05, 0.9])],
        # option id=4: two matches, source 1-2 from cluster 0, source 3 from cluster 2
        [np.array([1.0, 0.0]), np.array([0.9, 0.05]), np.array([0.9, 0.05]), np.array([0.05, 0.05])],
        # option id=5: all matches
        [np.array([1.0, 0.0]), np.array([0.9, 0.05]), np.array([0.9, 0.05]), np.array([0.9, 0.05])],
        # 2022-03-29 updates:
        # option id=6: option 2 variant, REDESIGN group 2 with latency shift
        [np.array([1.0, 0.0]), np.array([0.8, 0.1]), np.array([0.1, 0.8]), np.array([0.1, 0.1])],
        # option id=7: option 2 variant, maintain original three clusters, use group 2 to generate new subject.
        [np.array([0.0, 0.0]), np.array([0.8, 0.1]), np.array([0.1, 0.8]), np.array([0.1, 0.1])],
        # 2022-03-30 updates:
        # option id=8: option 2 variant, examine the noise effect, keep the mean structure the same
        [np.array([1.0, 0.0]), np.array([0.8, 0.1]), np.array([0.1, 0.8]), np.array([0.1, 0.1])]
    ]
    label_N_dict = generate_label_N_K_general(prob_ls_enum[prob_0_option_id], K)
else:
    prob_ls_enum = [
        # 2022-04-11 updates:
        # option id=9, N=9, K=4, use data from group 0 to generate subject 0,
        # the remaining 8 are evenly split into four groups
        [np.array([1.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
         np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
         np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
        # option id=10, N=9, K=4, use data from group 1 to generate subject 0,
        # the remaining 8 are evenly split into four groups
        [np.array([0.0, 1.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
         np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
         np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
        # option id=11, N=9, K=4, use data from group 2 to generate subject 0,
        # the remaining 8 are evenly split into four groups
        [np.array([0.0, 0.0, 1.0]),
         np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
         np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
         np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
        # option id=12, N=9, K=4, use data from group 3 to generate subject 0,
        # the remaining 8 are evenly split into four groups
        [np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
         np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
         np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
    ]
    label_N_dict = generate_label_N_K_general(prob_ls_enum[prob_0_option_id-9], K)
print(label_N_dict)


# Scenario 1:
# Try K114 channels Cz and PO7 as source
# Start with single-subject simulation.
# We assume different channels correspond to different people.
# The covariance matrix is independent for now.
output_dir = '{}/numpyro_output'.format(parent_dir)
source_dir = '{}/eeg_signal_source'.format(parent_dir)
source_k114_dir = '{}/MCMC/K114_mcmc_source.mat'.format(source_dir)

real_mcmc = sio.loadmat(source_k114_dir)
print(real_mcmc.keys())
real_beta_tar = real_mcmc['beta_tar']
real_beta_ntar = real_mcmc['beta_ntar']
Cz_id = channel_name_short.index('Cz')
PO7_id = channel_name_short.index('PO7')
# for the third cluster, provide the confusion purpose.
Fz_id = channel_name_short.index('Fz')

real_beta_tar_mean = np.mean(real_beta_tar, axis=(0, 3))
real_beta_ntar_mean = np.mean(real_beta_ntar, axis=(0, 3))
signal_length = real_beta_tar_mean.shape[1]

param_true_dict = {
    'target_char': target_char,
    'N': N,
    'seq_size': [seq_size for i in range(N)],
    'signal_length': signal_length,
    'label_N': label_N_dict,
    'group_0': {'beta_tar': real_beta_tar_mean[Cz_id, :]*3,
                'beta_ntar': real_beta_ntar_mean[Cz_id, :]*2,
                'sigma': sigma_val,
                'rho': rho_val
                },
    'group_1': {'beta_tar': real_beta_tar_mean[PO7_id, :],
                'beta_ntar': real_beta_ntar_mean[PO7_id, :],
                'sigma': sigma_val,
                'rho': rho_val
                },
    'group_2': {'beta_tar': real_beta_tar_mean[Fz_id, :],
                'beta_ntar': real_beta_ntar_mean[Fz_id, :],
                'sigma': sigma_val,
                'rho': rho_val}
}

if K == 3 and prob_0_option_id == 6:
    # only for option 6 where we consider the latency shift.
    param_true_dict['group_2']['beta_tar'] = param_true_dict['group_0']['beta_tar'][::-1]
    param_true_dict['group_2']['beta_ntar'] = param_true_dict['group_0']['beta_ntar'][::-1]

if K == 3 and prob_0_option_id == 7:
    # only for option 7 where we generate new subject data with group 2.
    # and enlarge the difference between target and non-target.
    # 2022-04-03 updates: I modify the sigma to be 3.0 to increase the SNR.
    param_true_dict['group_2']['beta_tar'] = np.copy(param_true_dict['group_0']['beta_tar']) * 0.5
    param_true_dict['group_2']['beta_ntar'] = np.copy(param_true_dict['group_0']['beta_ntar']) * 0.5

if K == 3 and prob_0_option_id == 8:
    # keep the mean structure the same and change the variance term.
    param_true_dict['group_1']['beta_tar'] = np.copy(param_true_dict['group_0']['beta_tar'])
    param_true_dict['group_1']['beta_ntar'] = np.copy(param_true_dict['group_0']['beta_ntar'])
    param_true_dict['group_1']['sigma'] = 8
    param_true_dict['group_1']['rho'] = 0.5

    param_true_dict['group_2']['beta_tar'] = np.copy(param_true_dict['group_0']['beta_tar'])
    param_true_dict['group_2']['beta_ntar'] = np.copy(param_true_dict['group_0']['beta_ntar'])
    param_true_dict['group_2']['sigma'] = 2
    param_true_dict['group_2']['rho'] = 0.5

if K == 4:
    param_true_dict['group_3'] = {
        'beta_tar': param_true_dict['group_0']['beta_tar'][::-1],
        'beta_ntar': param_true_dict['group_0']['beta_ntar'][::-1],
        'sigma': sigma_val,
        'rho': rho_val
    }


# produce the true ERP function plots (single-channel only)
if K == 3:
    fig0, ax0 = plt.subplots(1, 3, figsize=(18, 8))
else:
    fig0, ax0 = plt.subplots(1, 4, figsize=(21, 6))
group_k_channel_ids = [Cz_id, PO7_id, Fz_id]
for k in range(K):
    group_k_name = 'group_{}'.format(k)
    ax0[k].plot(
        np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_tar'],
        label='target', color='red'
    )
    ax0[k].plot(
        np.arange(param_true_dict['signal_length']), param_true_dict[group_k_name]['beta_ntar'],
        label='non-target', color='blue'
    )
    ax0[k].legend(loc='lower right')
    ax0[k].set_xlabel('Time (unit)')
    ax0[k].set_ylim((-6, 6))
    ax0[k].set_title('True curve, {}'.format(group_k_name))
# fig0.show()


target_char_test = list('THE0QUICK0BROWN0FOX')
seq_size_test = 10

scenario_name_dir = '{}/N_{}_K_{}'.format(output_dir, N, K)

if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

scenario_name = 'N_{}_K_{}_K114_based_option_{}_sigma_{}_rho_{}'.format(
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
    sim_data_dict = generate_simulated_data_signal_based(
        param_true_dict, sim_rcp_array, scenario_name_dir_try, seed_num=iter_num
    )

    # examine the simulated training data
    if N == 4:
        fig1, ax1 = plt.subplots(2, 2, figsize=(12, 12))
    else:
        fig1, ax1 = plt.subplots(3, 3, figsize=(16, 16))
    for i in range(N):
        subject_i_name = 'subject_{}'.format(i)
        sim_data_dict[subject_i_name]['X'] = np.squeeze(np.array([sim_data_dict[subject_i_name]['X']]), axis=0)

        mean_tar_i = np.mean(sim_data_dict[subject_i_name]['X'][:, [3, 7], :], axis=(0, 1))
        mean_ntar_i = np.mean(np.delete(sim_data_dict[subject_i_name]['X'], [3, 7], axis=1), axis=(0, 1))

        if N == 4:
            ax1_row = int(i / 2)
            ax1_col = i % 2
        else:
            ax1_row = int(i / 3)
            ax1_col = i % 3
        ax1[ax1_row, ax1_col].plot(np.arange(param_true_dict['signal_length']), mean_tar_i, label='target', color='red')
        ax1[ax1_row, ax1_col].plot(np.arange(param_true_dict['signal_length']), mean_ntar_i, label='non-target', color='blue')
        ax1[ax1_row, ax1_col].legend(loc='lower right')
        ax1[ax1_row, ax1_col].set_xlabel('Time (unit)')
        ax1[ax1_row, ax1_col].set_ylim([-8, 8])
        # ax1[0].set_ylim([-5, 6])
        ax1[ax1_row, ax1_col].set_title('{}, cluster_{}'.format(
            subject_i_name, sim_data_dict[subject_i_name]['label'])
        )
    fig1.savefig('{}/plot_sim_data_mean.png'.format(scenario_name_dir_try))

    # generate the testing data
    sim_data_test_dict = generate_simulated_test_data_signal_based(
        param_true_dict,  sim_data_dict['subject_0']['label'],  # new_subject_label
        target_char_test, seq_size_test, sim_rcp_array,
        scenario_name_dir_try, seed_num=iter_num
    )
