from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.EEGGlobal import *
plt.style.use("bmh")

# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
# select_channel_ids = np.array([15, 6]) - 1
select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)

if E_sub == E_total:
    select_channel_ids_str = 'all'

if local_use:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_local)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_local)
    sub_new_name = 'K145'
    seq_source = 5

else:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_slurm)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_source = int(sys.argv[2])

seq_size_train = 15
if sub_new_name in ['K154', 'K190']:
    seq_size_train = 20

sub_new_trn_dir = '{}/{}'.format(parent_trn_dir, sub_new_name)
trn_dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
sub_new_trn_dat_dir = '{}/{}_{}.mat'.format(sub_new_trn_dir, sub_new_name, trn_dat_name_common)

sub_new_test_dir = '{}/{}'.format(parent_frt_dir, sub_new_name)
test_dat_name_common_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
if sub_new_name in FRT_file_name_dict.keys():
    test_dat_name_common_ls = FRT_file_name_dict[sub_new_name]

# create new folder (e.g., FRT_files/K106/001_BCI_FRT/swLDA/channel_6/ (seq_i)...)
for test_dat_name_common in test_dat_name_common_ls:
    sub_new_test_dir_2 = '{}/{}'.format(sub_new_test_dir, test_dat_name_common)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_3 = '{}/swLDA'.format(sub_new_test_dir_2)
    if not os.path.exists('{}'.format(sub_new_test_dir_3)):
        os.mkdir(sub_new_test_dir_3)
    sub_new_test_dir_4 = '{}/channel_{}'.format(sub_new_test_dir_3, select_channel_ids_str)
    if not os.path.exists('{}'.format(sub_new_test_dir_4)):
        os.mkdir(sub_new_test_dir_4)


for seq_i in np.array([0,1,2,3,4,9]):
    print(seq_i + 1)
    # import swLDA output files
    swlda_output_dict = sio.loadmat('{}/swLDA/channel_{}/swLDA_output_seq_size_{}.mat'.format(
        sub_new_trn_dir, select_channel_ids_str, seq_i + 1)
    )

    # import training set (up to seq_i+1)
    signal_train_i, label_train_i, code_train_i = import_eeg_train_data(
        sub_new_trn_dat_dir, seq_i, select_channel_ids, E_total, signal_length, target_num_train, seq_size_train,
        reshape_2d_bool=True
    )

    swlda_prob_train_dict = {'NewOnly': {}, 'Mixture': {}}
    swlda_prob_test_dict = {'NewOnly': {}, 'Mixture': {}}

    for decision_rule in decision_rule_ls:
        b_inmodel_rule = swlda_output_dict['b{}'.format(decision_rule)]
        mu_rule_tar = swlda_output_dict['Mean{}Tar'.format(decision_rule)][0, 0]
        mu_rule_ntar = swlda_output_dict['Mean{}Ntar'.format(decision_rule)][0, 0]
        std_rule = swlda_output_dict['Std{}'.format(decision_rule)][0, 0]
        print(mu_rule_tar, mu_rule_ntar, std_rule)

        pred_score_train_0, pred_binary_train = swlda_predict_binary_likelihood(
            b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule, signal_train_i
        )
        pred_prob_letter_train, pred_prob_mat_train = ml_predict_letter_likelihood(
            pred_score_train_0, code_train_i, target_char_size,
            seq_i + 1, mu_rule_tar, mu_rule_ntar, std_rule,
            stimulus_group_set, sim_rcp_array
        )
        swlda_prob_train_dict[decision_rule] = {
            'letter': pred_prob_letter_train.tolist(),
            'prob': pred_prob_mat_train.tolist()
        }

        # testing set
        for test_dat_name_common in test_dat_name_common_ls:
            test_data_dir = '{}/{}_{}_Truncated_Data_0.5_6.mat'.format(
                sub_new_test_dir, sub_new_name, test_dat_name_common
            )
            signal_test_dict = import_eeg_test_data(
                test_data_dir, select_channel_ids, E_total, signal_length, reshape_2d_bool=True
            )
            signal_test = signal_test_dict['Signal']
            code_test = signal_test_dict['Code']
            seq_size_test = signal_test_dict['Seq_size']
            target_num_test = signal_test_dict['Text_size']

            pred_score_test_0, pred_binary_test = swlda_predict_binary_likelihood(
                b_inmodel_rule, mu_rule_tar, mu_rule_ntar, std_rule, signal_test
            )
            pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
                pred_score_test_0, code_test, target_num_test,
                seq_size_test, mu_rule_tar, mu_rule_ntar, std_rule,
                stimulus_group_set, sim_rcp_array
            )
            swlda_prob_test_dict[decision_rule][test_dat_name_common] = {
                'letter': pred_prob_letter_test.tolist(),
                'prob': pred_prob_mat_test.tolist()
            }

    swlda_predict_train_name_seq_i = 'swLDA_predict_train_seq_size_{}'.format(seq_i + 1)
    swlda_predict_train_name_dir_seq_i = '{}/swLDA/channel_{}/{}.json'.format(
        sub_new_trn_dir, select_channel_ids_str, swlda_predict_train_name_seq_i
    )
    with open(swlda_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(swlda_prob_train_dict, write_file)

    for test_dat_name_common in test_dat_name_common_ls:
        swlda_predict_test_name_seq_i = 'swLDA_predict_test_seq_size_{}'.format(seq_i + 1)
        swlda_predict_test_name_dir_seq_i = '{}/{}/swLDA/channel_{}/{}.json'.format(
            sub_new_test_dir, test_dat_name_common, select_channel_ids_str, swlda_predict_test_name_seq_i
        )
        swlda_prob_test_dat_dict = {
            'NewOnly': swlda_prob_test_dict['NewOnly'][test_dat_name_common]
            # 'Mixture': swlda_prob_test_dict['Mixture'][test_dat_name_common]
        }
        with open(swlda_predict_test_name_dir_seq_i, "w") as write_file:
            json.dump(swlda_prob_test_dat_dict, write_file)


