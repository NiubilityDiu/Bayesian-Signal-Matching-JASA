from self_py_fun.MCMCFun import *
from self_py_fun.ExistMLFun import *
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
E = 1
tol = 1e-3
domain_tradeoff = 0.5

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N = 7
    K = 3
    prob_0_option_id = 2
    sigma_val = 3.0
    rho_val = 0.6

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    sigma_val = float(sys.argv[5])
    rho_val = float(sys.argv[6])

seq_size_ls = [seq_source_size for i in range(N)]
scenario_name = 'N_{}_K_{}_option_{}_sigma_{}_rho_{}'.format(N, K, prob_0_option_id, sigma_val, rho_val)
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)

mdwm_dir = '{}/MDWM'.format(scenario_name_dir)
if not os.path.exists(mdwm_dir):
    os.mkdir(mdwm_dir)

# import training files
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)
with open(sim_data_dir, 'r') as file0:
    sim_data_dict = json.load(file0)

target_char_train = sim_data_dict['omega']
code_train = np.array(sim_data_dict['subject_0']['W'])
label_train = np.array(sim_data_dict['subject_0']['Y'])
signal_train = np.array(sim_data_dict['subject_0']['X'])

source_sub_name_ls = []
for i in range(N-1):
    source_sub_name_ls.append('subject_{}'.format(i+1))

# import the test data file
sim_data_test_name = 'sim_dat_test'
sim_data_test_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_test_name
)

with open(sim_data_test_dir, 'r') as file:
    sim_data_test_dict = json.load(file)

target_char_test = sim_data_test_dict['omega']
code_test = np.array(sim_data_test_dict['W']).astype('int8')
label_test = np.array(sim_data_test_dict['Y'])
signal_test = np.array(sim_data_test_dict['X'])
target_num_test, seq_num_test, _, _ = signal_test.shape
signal_test = np.reshape(
    signal_test, [target_num_test * seq_num_test * rcp_unit_flash_num, E, signal_length]
)


# vary the training seq size
for seq_i in np.arange(seq_source_size):
    print(seq_i + 1)
    # train mdwm:
    mdwm_obj, mdwm_ERP_cov_obj, X_domain_mdwm_size, mdwm_param_dict = mdwm_fit_from_sim_feature(
        sim_data_dict, 'subject_0', source_sub_name_ls, domain_tradeoff, tol, seq_size_train=seq_i+1, E=E
    )

    ############################
    ## Prediction on TRN file ##
    ############################
    mu_tar_mdwm = mdwm_param_dict['mu_tar']
    mu_ntar_mdwm = mdwm_param_dict['mu_ntar']
    std_common_mdwm = mdwm_param_dict['std_common']

    signal_train_i = np.reshape(
        signal_train[:, :(seq_i+1), ...], [target_char_size * (seq_i + 1) * rcp_unit_flash_num, E, signal_length]
    )
    code_train_i = np.reshape(
        code_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    ).astype('int8')
    label_train_i = np.reshape(
        label_train[:, :(seq_i+1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
    )

    signal_train_cov = mdwm_ERP_cov_obj.transform(signal_train_i) + np.eye(X_domain_mdwm_size) * tol
    pred_score_train_0 = np.log(mdwm_obj.predict_proba(signal_train_cov)[:, 0])
    pred_binary_train_0 = mdwm_obj.predict(signal_train_cov)
    pred_prob_letter_train, pred_prob_mat_train = ml_predict_letter_likelihood(
        pred_score_train_0, code_train_i, len(target_char_train),
        seq_i + 1, mu_tar_mdwm, mu_ntar_mdwm, std_common_mdwm,
        stimulus_group_set, sim_rcp_array
    )
    mdwm_prob_train_dict = {
        'letter': pred_prob_letter_train.tolist(),
        'prob': pred_prob_mat_train.tolist()
    }

    mdwm_predict_train_name_seq_i = 'predict_sub_0_train_seq_size_{}_MDWM'.format(seq_i + 1)
    mdwm_predict_train_name_dir_seq_i = '{}/MDWM/{}.json'.format(
        scenario_name_dir, mdwm_predict_train_name_seq_i
    )
    with open(mdwm_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(mdwm_prob_train_dict, write_file)


    ############################
    ## Prediction on FRT file ##
    ############################
    signal_test_cov = mdwm_ERP_cov_obj.transform(signal_test) + np.eye(X_domain_mdwm_size) * tol
    pred_score_test_0 = np.log(mdwm_obj.predict_proba(signal_test_cov)[:, 0])
    pred_binary_test_0 = mdwm_obj.predict(signal_test_cov)

    pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
        pred_score_test_0, code_test, target_num_test,
        seq_num_test, mu_tar_mdwm, mu_ntar_mdwm, std_common_mdwm,
        stimulus_group_set, sim_rcp_array
    )
    mdwm_prob_test_dict = {
        'letter': pred_prob_letter_test.tolist(),
        'prob': pred_prob_mat_test.tolist()
    }

    mdwm_predict_test_name_seq_i = 'predict_sub_0_test_seq_size_{}_MDWM'.format(seq_i + 1)
    mdwm_predict_test_name_dir_seq_i = '{}/MDWM/{}.json'.format(
        scenario_name_dir, mdwm_predict_test_name_seq_i
    )

    with open(mdwm_predict_test_name_dir_seq_i, "w") as write_file:
        json.dump(mdwm_prob_test_dict, write_file)



