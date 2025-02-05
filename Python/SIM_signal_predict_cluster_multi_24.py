from self_py_fun.MCMCMultiFun import *
plt.style.use("bmh")
import scipy.io as sio


E = 2
parent_path_sim_dir  = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/SIM_files'.format(parent_path_sim_dir)

iter_num = 0
N_total = 24
K = 24
log_lhd_diff_approx = 2.0

seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_multi_xdawn_eeg'.format(N_total, K)
scenario_name_dir = '{}/{}'.format(parent_data_dir, scenario_name)
scenario_iter_name_dir = '{}/iter_{}'.format(scenario_name_dir, iter_num)

sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_iter_name_dir, sim_data_name
)

cluster_name_dir = '{}/borrow_gibbs'.format(scenario_iter_name_dir)
if not os.path.exists('{}'.format(cluster_name_dir)):
    os.mkdir(cluster_name_dir)

# create the kernel matrix and related hyper-parameters
length_ls_2 = [[0.3, 0.2]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
eigen_val_dict = eigen_val_dict['group_0']
eigen_fun_mat_dict = eigen_fun_mat_dict['group_0']
eigen_fun_mat_tar = eigen_fun_mat_dict['target']

sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_iter_name_dir, sim_data_name
)
with open(sim_data_dir, 'r') as file0:
    sim_data_dict = json.load(file0)

target_char_train = sim_data_dict['omega']
code_train = np.array(sim_data_dict['subject_0']['W'])
label_train = np.array(sim_data_dict['subject_0']['Y'])
signal_train = np.array(sim_data_dict['subject_0']['X'])

# import the test data file
sim_data_test_name = 'sim_dat_test'
sim_data_test_dir = '{}/{}.json'.format(
    scenario_iter_name_dir, sim_data_test_name
)

with open(sim_data_test_dir, 'r') as file:
    sim_data_test_dict = json.load(file)

target_char_test = sim_data_test_dict['omega']
code_test = np.array(sim_data_test_dict['W']).astype('int8')
label_test = np.array(sim_data_test_dict['Y'])
signal_test = np.array(sim_data_test_dict['X'])
target_num_test, seq_num_test, _, _, _ = signal_test.shape
signal_test_rs = np.reshape(
    signal_test, [target_num_test * seq_num_test * rcp_unit_flash_num, E, signal_length]
)

for seq_i in np.array([1,2,3,4]):
    print(seq_i + 1)

    signal_train_i = np.reshape(
        signal_train[:, :(seq_i+1), ...], [target_char_size * (seq_i + 1) * rcp_unit_flash_num, E, signal_length]
    )
    code_train_i = np.reshape(
        code_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    ).astype('int8')
    label_train_i = np.reshape(
        label_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    )

    # import mcmc output files
    mcmc_output_dict_dir = '{}/mcmc_sub_0_seq_size_{}_cluster_log_lhd_diff_approx_{}.mat'.format(
        cluster_name_dir, seq_i+1, log_lhd_diff_approx)
    mcmc_output_dict = sio.loadmat(mcmc_output_dict_dir, simplify_cells=True)
    mcmc_ids_seq_i = None

    pred_score_train_0_i, _, _ = predict_cluster_fast_multi(
        mcmc_output_dict, E, signal_length, signal_train_i, mcmc_ids_seq_i
    )
    mu_tar_cluster = np.mean(pred_score_train_0_i[label_train_i == 1])
    mu_ntar_cluster = np.mean(pred_score_train_0_i[label_train_i != 1])
    std_common_cluster = np.std(pred_score_train_0_i)
    print('{}, {}, {}'.format(mu_tar_cluster, mu_ntar_cluster, std_common_cluster))

    # training set
    pred_prob_letter, pred_prob_mat = ml_predict_letter_likelihood(
        pred_score_train_0_i, code_train_i, target_char_size,
        seq_i + 1, mu_tar_cluster, mu_ntar_cluster, std_common_cluster,
        stimulus_group_set, sim_rcp_array
    )
    cluster_prob_train_dict = {
        'letter': pred_prob_letter.tolist(),
        'prob': pred_prob_mat.tolist(),
    }
    cluster_predict_train_name_seq_i = 'predict_sub_0_train_seq_size_{}_cluster'.format(seq_i + 1)
    cluster_predict_train_name_dir_seq_i = '{}/{}.json'.format(
        cluster_name_dir, cluster_predict_train_name_seq_i
    )
    with open(cluster_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(cluster_prob_train_dict, write_file)

    # testing set
    pred_score_test_0, _, _ = predict_cluster_fast_multi(
        mcmc_output_dict, E, signal_length, signal_test, mcmc_ids_seq_i
    )
    pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
        pred_score_test_0, code_test, target_num_test,
        seq_num_test, mu_tar_cluster, mu_ntar_cluster, std_common_cluster,
        stimulus_group_set, sim_rcp_array
    )

    cluster_prob_test_dict = {
        'letter': pred_prob_letter_test.tolist(),
        'prob': pred_prob_mat_test.tolist(),
    }

    cluster_predict_test_name_seq_i = 'predict_sub_0_test_seq_size_{}_cluster'.format(seq_i + 1)
    cluster_predict_test_name_dir_seq_i = '{}/{}.json'.format(cluster_name_dir, cluster_predict_test_name_seq_i)
    with open(cluster_predict_test_name_dir_seq_i, "w") as write_file:
        json.dump(cluster_prob_test_dict, write_file)
