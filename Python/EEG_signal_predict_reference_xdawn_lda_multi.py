from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.EEGGlobal import *
plt.style.use("bmh")

# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)

if local_use:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_local)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_local)
    sub_new_name = 'K145'
    seq_source = 5
    n_components = 2

else:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_slurm)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_source = int(sys.argv[2])
    n_components = int(sys.argv[3])

seq_size_train = 15
if sub_new_name in ['K154', 'K190']:
    seq_size_train = 20

sub_new_trn_dir = '{}/{}'.format(parent_trn_dir, sub_new_name)
trn_dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
ref_name = 'reference_numpyro_letter_{}_xdawn_lda'.format(letter_dim_sub)
sub_new_trn_dat_dir = '{}/{}_{}.mat'.format(sub_new_trn_dir, sub_new_name, trn_dat_name_common)
sub_new_ref_dir = '{}/{}'.format(sub_new_trn_dir, ref_name)
if not os.path.exists(sub_new_ref_dir):
    os.mkdir(sub_new_ref_dir)
sub_new_ref_dir_2 = '{}/channel_all_comp_{}'.format(sub_new_ref_dir, n_components)
if not os.path.exists(sub_new_ref_dir_2):
    os.mkdir(sub_new_ref_dir_2)
xdawn_min = min(E_sub, n_components)

sub_new_test_dir = '{}/{}'.format(parent_frt_dir, sub_new_name)
test_dat_name_common_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
if sub_new_name in FRT_file_name_dict.keys():
    test_dat_name_common_ls = FRT_file_name_dict[sub_new_name]

# create new folder (e.g., FRT_files/K106/001_BCI_FRT/numpyro_reference/channel_6/
for test_dat_name_common in test_dat_name_common_ls:
    sub_new_test_dir_2 = '{}/{}'.format(sub_new_test_dir, test_dat_name_common)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_3 = '{}/{}'.format(sub_new_test_dir_2, ref_name)
    if not os.path.exists('{}'.format(sub_new_test_dir_3)):
        os.mkdir(sub_new_test_dir_3)
    sub_new_test_dir_4 = '{}/channel_all_comp_{}'.format(sub_new_test_dir_3, n_components)
    if not os.path.exists('{}'.format(sub_new_test_dir_4)):
        os.mkdir(sub_new_test_dir_4)

for seq_i in np.array([0,1,2,3,4,9]):
    print(seq_i + 1)
    # import training set (up to seq_i+1)
    signal_train_i, label_train_i, code_train_i = import_eeg_train_data(
        sub_new_trn_dat_dir, seq_i, select_channel_ids, E_total,
        signal_length, target_num_train, seq_size_train,
        reshape_2d_bool=False
    )

    [xdawn_lda_prob_train_dict, xdawn_lda_train_obj, mu_tar_xdawn_lda,
     mu_ntar_xdawn_lda, std_common_xdawn_lda] = predict_char_accuracy_multi(
        signal_train_i, label_train_i, code_train_i, None,
        seq_i + 1, xdawn_min, signal_length, None,
        target_char_size, 'xdawn_lda',
        n_components=n_components, xdawn_lda_obj=None
    )
    xdawn_lda_predict_train_name_seq_i = 'reference_xdawn_lda_predict_train_seq_size_{}'.format(seq_i + 1)
    xdawn_lda_predict_train_name_dir_seq_i = '{}/{}.json'.format(
        sub_new_ref_dir_2, xdawn_lda_predict_train_name_seq_i
    )
    with open(xdawn_lda_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(xdawn_lda_prob_train_dict, write_file)

    # testing set
    for test_dat_name_common in test_dat_name_common_ls:
        test_data_dir = '{}/{}_{}_Truncated_Data_0.5_6.mat'.format(
            sub_new_test_dir, sub_new_name, test_dat_name_common
        )
        signal_test_dict = import_eeg_test_data(
            test_data_dir, select_channel_ids, E_total, signal_length, reshape_2d_bool=False
        )
        signal_test = signal_test_dict['Signal']
        code_test = signal_test_dict['Code']
        seq_size_test = signal_test_dict['Seq_size']
        target_num_test = signal_test_dict['Text_size']

        [xdawn_lda_prob_test_dict, _, _, _, _] = predict_char_accuracy_multi(
            signal_test, None, code_test, None,
            seq_size_test, xdawn_min, signal_length, None,
            target_num_test, 'xdawn_lda',
            n_components=n_components, xdawn_lda_obj=xdawn_lda_train_obj,
            mu_tar=mu_tar_xdawn_lda, mu_ntar=mu_ntar_xdawn_lda,
            std_common=std_common_xdawn_lda
        )
        xdawn_lda_predict_test_name_seq_i = 'reference_xdawn_lda_predict_test_seq_size_{}'.format(seq_i + 1)
        xdawn_lda_predict_test_name_dir_seq_i = '{}/{}/{}/channel_all_comp_{}/{}.json'.format(
            sub_new_test_dir, test_dat_name_common, ref_name, n_components,
            xdawn_lda_predict_test_name_seq_i
        )
        with open(xdawn_lda_predict_test_name_dir_seq_i, "w") as write_file:
            json.dump(xdawn_lda_prob_test_dict, write_file)

