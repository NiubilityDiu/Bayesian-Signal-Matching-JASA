import sys
import matplotlib.pyplot as plt
from self_py_fun.SimFun import *
plt.style.use("bmh")
assert numpyro.__version__.startswith("0.9.0")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
seq_size = 10
prob_threshold = 0.5

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N = 9
    K = 4
    prob_0_option_id = 0
    sigma_val = 3.0
    rho_val = 0.5

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])

scenario_name = 'N_{}_K_{}_K114_based_option_{}_sigma_{}_rho_{}'.format(
    N, K, prob_0_option_id, sigma_val, rho_val
)
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N, K, scenario_name, iter_num)
decision_rule_ls = ['NewOnly', 'Strict', 'Flexible']


for seq_i in range(seq_size):
    # import swLDA output files.
    swlda_output_name = 'swLDA_output_seq_size_{}_threshold_{}'.format(seq_i+1, prob_threshold)
    swlda_folder_dir = '{}/swLDA/{}.mat'.format(scenario_name_dir, swlda_output_name)
    swlda_output_dict = sio.loadmat(swlda_folder_dir)
    # print the state
    print(np.transpose(swlda_output_dict['SourceSubState']))


    # Scenario 1:
    # import the training data file (prediction on training files)
    sim_data_name = 'sim_dat'
    sim_data_dir = '{}/{}.json'.format(
        scenario_name_dir, sim_data_name
    )
    with open(sim_data_dir, 'r') as file0:
        sim_data_dict = json.load(file0)

    target_char_train = sim_data_dict['omega']
    target_num_train = len(target_char_train)
    code_train = np.array(sim_data_dict['subject_0']['W'])
    label_train = np.array(sim_data_dict['subject_0']['Y'])
    signal_train = np.array(sim_data_dict['subject_0']['X'])
    seq_num_train, _, signal_length = signal_train.shape

    swlda_prob_train_dict = {'NewOnly': {}, 'Strict': {}, 'Flexible': {}}
    for decision_rule in decision_rule_ls:
        b_inmodel_rule = swlda_output_dict['b{}'.format(decision_rule)]
        mu_rule_tar = swlda_output_dict['Mean{}Tar'.format(decision_rule)][0, 0]
        mu_rule_ntar = swlda_output_dict['Mean{}Ntar'.format(decision_rule)][0, 0]
        std_rule = swlda_output_dict['Std{}'.format(decision_rule)][0, 0]

        result_prob_letter, result_prob_mat = summarize_accuracy_prob(
            b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule,
            np.reshape(signal_train[:seq_i + 1, ...], [target_num_train * (seq_i + 1) * rcp_unit_flash_num, signal_length]),
            np.reshape(code_train[:seq_i + 1, :].astype('int8'), [target_num_train * (seq_i + 1) * rcp_unit_flash_num, 1]),
            target_char_train, seq_i + 1,
            stimulus_group_set, sim_rcp_array
        )

        swlda_prob_train_dict[decision_rule]['seq_train_{}'.format(seq_i + 1)] = \
            result_prob_mat[0, :, sim_rcp_array.index(target_char_train)].tolist()

    # save the prob by training sizes
    swlda_predict_train_name_seq_i = 'swLDA_predict_train_seq_size_{}_threshold_{}'.format(seq_i+1, prob_threshold)
    swlda_predict_train_name_dir_seq_i = '{}/swLDA/{}.json'.format(scenario_name_dir, swlda_predict_train_name_seq_i)
    with open(swlda_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(swlda_prob_train_dict, write_file)

    # import the test data file
    sim_data_test_name = 'sim_dat_test'
    sim_data_test_dir = '{}/{}.json'.format(
        scenario_name_dir, sim_data_test_name
    )
    
    with open(sim_data_test_dir, 'r') as file:
        sim_data_test_dict = json.load(file)
    
    target_char_test = sim_data_test_dict['omega']
    code_test = np.array(sim_data_test_dict['W'])
    label_test = np.array(sim_data_test_dict['Y'])
    signal_test = np.array(sim_data_test_dict['X'])
    target_num_test, seq_num_test, _, _ = signal_test.shape
    
    # baseline approach, new only
    swlda_predict_dict_seq_i = {'NewOnly': {}, 'Strict': {}, 'Flexible': {}}
    for decision_rule in decision_rule_ls:
        b_inmodel_rule = swlda_output_dict['b{}'.format(decision_rule)]
        mu_rule_tar = swlda_output_dict['Mean{}Tar'.format(decision_rule)][0, 0]
        mu_rule_ntar = swlda_output_dict['Mean{}Ntar'.format(decision_rule)][0, 0]
        std_rule = swlda_output_dict['Std{}'.format(decision_rule)][0, 0]
    
        for seq_num_test_i in range(seq_num_test):
            [score_rule_test_iter, binary_rule_test_iter, result_rule_dict_test_iter] = summarize_accuracy_likelihood(
                b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule,
                np.reshape(signal_test[:, :seq_num_test_i + 1, ...], [target_num_test * (seq_num_test_i + 1) * rcp_unit_flash_num, signal_length]),
                np.reshape((label_test[:, :seq_num_test_i + 1, :] - 0.5) * 2, [target_num_test * (seq_num_test_i + 1) * rcp_unit_flash_num, 1]),
                np.reshape(code_test[:, :seq_num_test_i + 1, :].astype('int8'), [target_num_test * (seq_num_test_i + 1) * rcp_unit_flash_num, 1]),
                'subject_0', target_char_test, seq_num_test_i + 1, seq_i + 1,
                stimulus_group_set, sim_rcp_array, decision_rule
            )
            print(result_rule_dict_test_iter)
            swlda_predict_dict_seq_i[decision_rule]['seq_test_{}'.format(seq_num_test_i + 1)] = result_rule_dict_test_iter
    
    # save the testing prediction by training size
    swlda_predict_name_seq_i = 'swLDA_predict_test_seq_size_{}_threshold_{}'.format(seq_i + 1, prob_threshold)
    swlda_predict_name_dir_seq_i = '{}/swLDA/{}.json'.format(scenario_name_dir, swlda_predict_name_seq_i)
    with open(swlda_predict_name_dir_seq_i, "w") as write_file:
        json.dump(swlda_predict_dict_seq_i, write_file)


