from self_py_fun.MCMCMultiFun import *
from pyriemann.estimation import ERPCovariances
from pyriemann.classification import FgMDM
from sklearn.pipeline import make_pipeline
plt.style.use("bmh")


local_use = True
# local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
target_char = 'TT'
target_char_size = len(target_char)
E = 2
matrix_normal_bool = True
signal_length = 35

if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N_total = 7
    K = 3
    prob_0_option_id = 1
    seq_i = 2

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N_total = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    seq_i = int(sys.argv[5])  # take values from 0 to 9

seq_size_ls = [seq_source_size for i in range(N_total)]

scenario_name = 'N_{}_K_{}_multi_option_{}'.format(
    N_total, K, prob_0_option_id
)
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N_total, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)
mdm_name_dir = '{}/reference_mdm'.format(scenario_name_dir, K)
if not os.path.exists('{}'.format(mdm_name_dir)):
    os.mkdir(mdm_name_dir)

# import sim_data
source_data, new_data = import_sim_data(sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length, matrix_normal_bool)

clf = make_pipeline(
    ERPCovariances(estimator='scm'),
    FgMDM(metric='riemann')
)

X_new_data_i = np.concatenate([new_data['target'], new_data['non-target']], axis=0)
y_new_data_i = np.concatenate([np.ones(target_char_size * (seq_i + 1) * 2),
                               np.zeros(target_char_size * (seq_i + 1) * 10)], axis=0)


clf.fit(X_new_data_i, y_new_data_i)


# prediction section:
# import training set
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
target_num_test, seq_num_test, _, _, _ = signal_test.shape
signal_test_rs = np.reshape(
    signal_test, [target_num_test * seq_num_test * rcp_unit_flash_num, E, signal_length]
)



signal_train_i = np.reshape(
    signal_train[:, :(seq_i + 1), ...], [target_char_size * (seq_i + 1) * rcp_unit_flash_num, E, signal_length]
)
code_train_i = np.reshape(
    code_train[:, :(seq_i + 1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
).astype('int8')
label_train_i = np.reshape(
    label_train[:, :(seq_i + 1), :], [target_char_size * (seq_i + 1) * rcp_unit_flash_num]
)

pred_score_train_0_i = np.log(clf.predict_proba(signal_train_i)[:, 1])
pred_binary_train_i = clf.predict(signal_train_i)
mu_tar_mdm = np.mean(pred_score_train_0_i[label_train_i == 1])
mu_ntar_mdm = np.mean(pred_score_train_0_i[label_train_i != 1])
std_common_mdm = np.std(pred_score_train_0_i)
print('{}, {}, {}'.format(mu_tar_mdm, mu_ntar_mdm, std_common_mdm))

# training set
mdm_prob_train_dict = {}
pred_prob_letter, pred_prob_mat = ml_predict_letter_likelihood(
    pred_score_train_0_i, code_train_i, target_char_size,
    seq_i + 1, mu_tar_mdm, mu_ntar_mdm, std_common_mdm,
    stimulus_group_set, sim_rcp_array
)
mdm_prob_train_dict['letter'] = pred_prob_letter.tolist()
mdm_prob_train_dict['prob'] = pred_prob_mat.tolist()

# testing set
mdm_prob_test_dict = {}
# pred_score_test_0, pred_binary_test = swlda_predict_binary_likelihood(
#     b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule, signal_test_rs
# )
pred_score_test_0 = np.log(clf.predict_proba(signal_test_rs)[:, 1])
pred_binary_test = clf.predict(signal_test_rs)
pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
    pred_score_test_0, code_test, target_num_test,
    seq_num_test, mu_tar_mdm, mu_ntar_mdm, std_common_mdm,
    stimulus_group_set, sim_rcp_array
)
mdm_prob_test_dict['letter'] = pred_prob_letter_test.tolist()
mdm_prob_test_dict['prob'] = pred_prob_mat_test.tolist()

mdm_predict_train_name_seq_i = 'predict_sub_0_train_seq_size_{}_mdm'.format(seq_i+1)
mdm_predict_train_name_dir_seq_i = '{}/{}.json'.format(mdm_name_dir, mdm_predict_train_name_seq_i)
with open(mdm_predict_train_name_dir_seq_i, "w") as write_file:
    json.dump(mdm_prob_train_dict, write_file)

mdm_predict_test_name_seq_i = 'predict_sub_0_test_seq_size_{}_mdm'.format(seq_i + 1)
mdm_predict_test_name_dir_seq_i = '{}/{}.json'.format(mdm_name_dir, mdm_predict_test_name_seq_i)
with open(mdm_predict_test_name_dir_seq_i, "w") as write_file:
    json.dump(mdm_prob_test_dict, write_file)





