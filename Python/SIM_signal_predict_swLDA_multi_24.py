from self_py_fun.MCMCMultiFun import *
plt.style.use("bmh")
import scipy.io as sio


E = 2
decision_rule_ls = ['NewOnly', 'Mixture']
parent_path_sim_dir  = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/SIM_files'.format(parent_path_sim_dir)

iter_num = 0
N_total = 24
K = 24

seq_size_ls = [seq_source_size for i in range(N_total)]
scenario_name = 'N_{}_K_{}_multi_xdawn_eeg'.format(N_total, K)
scenario_name_dir = '{}/{}'.format(parent_data_dir, scenario_name)
scenario_iter_name_dir = '{}/iter_{}'.format(scenario_name_dir, iter_num)

sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_iter_name_dir, sim_data_name
)

swlda_dir = '{}/swLDA'.format(scenario_iter_name_dir)
if not os.path.exists(swlda_dir):
    os.mkdir(swlda_dir)

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
    signal_test,
    [target_num_test * seq_num_test * rcp_unit_flash_num, E * signal_length]
)

for seq_i in np.array([1,2,3,4]):
    print('Training sequence size: {}'.format(seq_i + 1))
    # import swLDA output files.
    swlda_output_name = 'swLDA_output_seq_size_{}'.format(seq_i + 1)
    swlda_folder_dir = '{}/{}.mat'.format(swlda_dir, swlda_output_name)
    swlda_output_dict = sio.loadmat(swlda_folder_dir)

    signal_train_i = np.reshape(
        signal_train[:, :(seq_i+1), ...], [target_char_size * (seq_i + 1) * rcp_unit_flash_num, E * signal_length]
    )
    code_train_i = np.reshape(
        code_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    ).astype('int8')
    label_train_i = np.reshape(
        label_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    )

    swlda_prob_train_dict = {'NewOnly': {}, 'Mixture': {}}
    swlda_prob_test_dict = {'NewOnly': {}, 'Mixture': {}}

    for decision_rule in decision_rule_ls:
        b_inmodel_rule = swlda_output_dict['b{}'.format(decision_rule)]
        mu_rule_tar = swlda_output_dict['Mean{}Tar'.format(decision_rule)][0, 0]
        mu_rule_ntar = swlda_output_dict['Mean{}Ntar'.format(decision_rule)][0, 0]
        std_rule = swlda_output_dict['Std{}'.format(decision_rule)][0, 0]
        print(mu_rule_tar, mu_rule_ntar, std_rule)

        # training set
        pred_score_train_0, pred_binary_train = swlda_predict_binary_likelihood(
            b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule, signal_train_i
        )
        pred_prob_letter_train, pred_prob_mat_train = ml_predict_letter_likelihood(
            pred_score_train_0, code_train_i, target_char_size,
            seq_i + 1, mu_rule_tar, mu_rule_ntar, std_rule,
            stimulus_group_set, sim_rcp_array
        )
        swlda_prob_train_dict[decision_rule]['letter'] = pred_prob_letter_train.tolist()
        swlda_prob_train_dict[decision_rule]['prob'] = pred_prob_mat_train.tolist()

        # testing set
        pred_score_test_0, pred_binary_test = swlda_predict_binary_likelihood(
            b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule, signal_test_rs
        )
        pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
            pred_score_test_0, code_test, target_num_test,
            seq_num_test, mu_rule_tar, mu_rule_ntar, std_rule,
            stimulus_group_set, sim_rcp_array
        )
        swlda_prob_test_dict[decision_rule]['letter'] = pred_prob_letter_test.tolist()
        swlda_prob_test_dict[decision_rule]['prob'] = pred_prob_mat_test.tolist()

    swlda_predict_train_name_seq_i = 'predict_sub_0_train_seq_size_{}_swLDA'.format(seq_i+1)
    swlda_predict_train_name_dir_seq_i = '{}/{}.json'.format(swlda_dir, swlda_predict_train_name_seq_i)
    with open(swlda_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(swlda_prob_train_dict, write_file)

    swlda_predict_test_name_seq_i = 'predict_sub_0_test_seq_size_{}_swLDA'.format(seq_i+1)
    swlda_predict_test_name_dir_seq_i = '{}/{}.json'.format(swlda_dir, swlda_predict_test_name_seq_i)
    with open(swlda_predict_test_name_dir_seq_i, "w") as write_file:
        json.dump(swlda_prob_test_dict, write_file)
