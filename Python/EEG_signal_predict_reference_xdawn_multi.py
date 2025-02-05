from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.EEGGlobal import *
import scipy.io as sio
plt.style.use("bmh")

# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)
hyper_param_bool = True

if local_use:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_local)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_local)
    sub_new_name = 'K145'
    seq_source = 10
    n_components = 2
    length_ls = [[0.3, 0.2]]
    gamma_val_ls = [[1.2, 1.2]]

else:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_slurm)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_source = int(sys.argv[2])
    n_components = int(sys.argv[3])
    # sensitivity analysis check
    length_ls = [[float(sys.argv[4]), float(sys.argv[5])]]
    gamma_val_ls = [[float(sys.argv[6]), float(sys.argv[6])]]

seq_size_train = 15
if sub_new_name in ['K154', 'K190']:
    seq_size_train = 20
channel_name = 'channel_all_comp_{}'.format(n_components)

sub_new_trn_dir = '{}/{}'.format(parent_trn_dir, sub_new_name)
trn_dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
ref_name = 'reference_numpyro_letter_{}_xdawn'.format(letter_dim_sub)
sub_new_trn_dat_dir = '{}/{}_{}.mat'.format(sub_new_trn_dir, sub_new_name, trn_dat_name_common)
sub_new_ref_dir = '{}/{}'.format(sub_new_trn_dir, ref_name)
sub_new_ref_dir_2 = '{}/{}'.format(sub_new_ref_dir, channel_name)
kernel_name = 'length_{}_{}_gamma_{}'.format(length_ls[0][0], length_ls[0][1], gamma_val_ls[0][0])
if hyper_param_bool:
    # sensitivity analysis check
    sub_new_ref_dir_2 = '{}/{}'.format(sub_new_ref_dir_2, kernel_name)
xdawn_min = min(E_sub, n_components)

# kernel hyper-parameters
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls, gamma_val_ls, 1, signal_length
)

sub_new_test_dir = '{}/{}'.format(parent_frt_dir, sub_new_name)
test_dat_name_common_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
if sub_new_name in FRT_file_name_dict.keys():
    test_dat_name_common_ls = FRT_file_name_dict[sub_new_name]

# create new folder (e.g., FRT_files/K106/001_BCI_FRT/numpyro_reference/channel_6/ (seq_i)...)
for test_dat_name_common in test_dat_name_common_ls:
    sub_new_test_dir_2 = '{}/{}'.format(sub_new_test_dir, test_dat_name_common)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_3 = '{}/{}'.format(sub_new_test_dir_2, ref_name)
    if not os.path.exists('{}'.format(sub_new_test_dir_3)):
        os.mkdir(sub_new_test_dir_3)
    sub_new_test_dir_4 = '{}/{}'.format(sub_new_test_dir_3, channel_name)
    if not os.path.exists('{}'.format(sub_new_test_dir_4)):
        os.mkdir(sub_new_test_dir_4)
    if hyper_param_bool:
        sub_new_test_dir_5 = '{}/{}'.format(sub_new_test_dir_4, kernel_name)
        if not os.path.exists('{}'.format(sub_new_test_dir_5)):
            os.mkdir(sub_new_test_dir_5)

for seq_i in np.array([1,2,3,4,9]):
    print(seq_i + 1)

    # import mcmc output files
    mcmc_output_dict = sio.loadmat('{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
        sub_new_ref_dir_2, seq_i + 1)
    )

    # import training set (up to seq_i+1)
    signal_train_i, label_train_i, code_train_i = import_eeg_train_data(
        sub_new_trn_dat_dir, seq_i, select_channel_ids, E_total, signal_length, target_num_train, seq_size_train,
        reshape_2d_bool=False
    )

    xdawn_filter_i_dir = '{}/{}/xdawn_filter_train_seq_size_{}.mat'.format(sub_new_ref_dir, channel_name, seq_i + 1)
    xdawn_filter_i = sio.loadmat(xdawn_filter_i_dir)['filter']
    signal_train_i = xdawn_filter_i[np.newaxis, ...] @ signal_train_i

    [ref_prob_train_dict, _, mu_tar_ref,
     mu_ntar_ref, std_common_ref] = predict_char_accuracy_multi(
        signal_train_i, label_train_i, code_train_i, mcmc_output_dict,
        seq_i + 1, xdawn_min, signal_length, eigen_fun_mat_dict['group_0'],
        target_char_size, 'ref'
    )

    ref_predict_train_name_seq_i = 'reference_xdawn_predict_train_seq_size_{}'.format(seq_i + 1)
    ref_predict_train_name_dir_seq_i = '{}/{}.json'.format(
        sub_new_ref_dir_2, ref_predict_train_name_seq_i
    )
    with open(ref_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(ref_prob_train_dict, write_file)

    # testing set
    for test_dat_name_common in test_dat_name_common_ls:
        test_data_dir = '{}/{}_{}_Truncated_Data_0.5_6.mat'.format(
            sub_new_test_dir, sub_new_name, test_dat_name_common
        )
        signal_test_dict = import_eeg_test_data(
            test_data_dir, select_channel_ids, E_total, signal_length, reshape_2d_bool=False
        )
        signal_test = signal_test_dict['Signal']

        signal_test = xdawn_filter_i[np.newaxis, ...] @ signal_test
        code_test = signal_test_dict['Code']
        seq_size_test = signal_test_dict['Seq_size']
        target_num_test = signal_test_dict['Text_size']

        [ref_prob_test_dict, _, _, _, _] = predict_char_accuracy_multi(
            signal_test, None, code_test, mcmc_output_dict,
            seq_size_test, xdawn_min, signal_length, eigen_fun_mat_dict['group_0'],
            target_num_test, 'ref',
            mu_tar=mu_tar_ref, mu_ntar=mu_ntar_ref, std_common=std_common_ref
        )

        ref_predict_test_name_seq_i = 'reference_xdawn_predict_test_seq_size_{}'.format(seq_i + 1)
        if hyper_param_bool:
            ref_predict_test_name_dir_seq_i = '{}/{}/{}/{}/{}/{}.json'.format(
                sub_new_test_dir, test_dat_name_common, ref_name, channel_name, kernel_name, ref_predict_test_name_seq_i
            )
        else:
            ref_predict_test_name_dir_seq_i = '{}/{}/{}/{}/{}.json'.format(
                sub_new_test_dir, test_dat_name_common, ref_name, channel_name, ref_predict_test_name_seq_i
            )
        with open(ref_predict_test_name_dir_seq_i, "w") as write_file:
            json.dump(ref_prob_test_dict, write_file)

