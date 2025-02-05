from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.EEGGlobal import *
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)
hyper_param_bool = True

if local_use:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_local)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_local)
    sub_new_name = 'K155'
    seq_source = 5
    n_components = 2
    reduced_bool = True
    length_ls = [[0.30, 0.20]]
    gamma_val_ls = [[1.2, 1.2]]
    log_lhd_approx = 1.0
    delta_threshold = 0.1

else:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_slurm)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_source = int(sys.argv[2])
    n_components = int(sys.argv[3])
    reduced_bool = (sys.argv[4] == 'T' or sys.argv[4] == 'True')
    # sensitivity analysis check
    length_ls = [[float(sys.argv[5]), float(sys.argv[6])]]
    gamma_val_ls = [[float(sys.argv[7]), float(sys.argv[7])]]
    log_lhd_approx = float(sys.argv[8])
    delta_threshold = float(sys.argv[9])

channel_name = 'channel_all_comp_{}'.format(n_components)
seq_size_train = 15
if sub_new_name in ['K154', 'K190']:
    seq_size_train = 20

sub_new_trn_dir = '{}/{}'.format(parent_trn_dir, sub_new_name)
trn_dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
sub_new_trn_dat_dir = '{}/{}_{}.mat'.format(sub_new_trn_dir, sub_new_name, trn_dat_name_common)
if reduced_bool:
    cluster_name = 'borrow_gibbs_letter_{}_reduced_xdawn'.format(letter_dim_sub)
    mixture_xdawn_name = 'mixture_gibbs_letter_{}_reduced_xdawn'.format(letter_dim_sub)
else:
    cluster_name = 'borrow_gibbs_letter_{}_xdawn'.format(letter_dim_sub)
    mixture_xdawn_name = 'mixture_gibbs_letter_{}_xdawn'.format(letter_dim_sub)

xdawn_min = min(E_sub, n_components)
ref_xdawn_name = 'reference_numpyro_letter_{}_xdawn'.format(letter_dim_sub)
kernel_name = 'length_{}_{}_gamma_{}'.format(length_ls[0][0], length_ls[0][1], gamma_val_ls[0][0])

# kernel hyper-parameters
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls, gamma_val_ls, 1, signal_length
)

sub_new_test_dir = '{}/{}'.format(parent_frt_dir, sub_new_name)
test_dat_name_common_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
if sub_new_name in FRT_file_name_dict.keys():
    test_dat_name_common_ls = FRT_file_name_dict[sub_new_name]

# create new folder (e.g., FRT_files/K106/001_BCI_FRT/N_41_K_5_gibbs/channel_6/ (seq_i)...)
mixture_xdawn_trn_dir = '{}/{}'.format(sub_new_trn_dir, mixture_xdawn_name)
if not os.path.exists(mixture_xdawn_trn_dir):
    os.mkdir(mixture_xdawn_trn_dir)
mixture_xdawn_trn_dir_2 = '{}/channel_all_comp_{}'.format(mixture_xdawn_trn_dir, n_components)
if not os.path.exists(mixture_xdawn_trn_dir_2):
    os.mkdir(mixture_xdawn_trn_dir_2)

for test_dat_name_common in test_dat_name_common_ls:
    sub_new_test_dir_2 = '{}/{}'.format(sub_new_test_dir, test_dat_name_common)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_3 = '{}/{}'.format(sub_new_test_dir_2, mixture_xdawn_name)
    if not os.path.exists('{}'.format(sub_new_test_dir_3)):
        os.mkdir(sub_new_test_dir_3)
    sub_new_test_dir_4 = '{}/{}'.format(sub_new_test_dir_3, channel_name)
    if not os.path.exists('{}'.format(sub_new_test_dir_4)):
        os.mkdir(sub_new_test_dir_4)
    if hyper_param_bool:
        sub_new_test_dir_5 = '{}/{}'.format(sub_new_test_dir_4, kernel_name)
        if not os.path.exists('{}'.format(sub_new_test_dir_5)):
            os.mkdir(sub_new_test_dir_5)


for seq_i in np.array([5]):
    print('Training Sequence of New Participant {}: {}'.format(sub_new_name, seq_i + 1))

    # import training set (up to seq_i+1)
    signal_train_i, label_train_i, code_train_i = import_eeg_train_data(
        sub_new_trn_dat_dir, seq_i, select_channel_ids, E_total, signal_length, target_num_train, seq_size_train,
        reshape_2d_bool=False
    )
    xdawn_filter_i_dir = '{}/{}/{}/xdawn_filter_train_seq_size_{}.mat'.format(
        sub_new_trn_dir, ref_xdawn_name, channel_name, seq_i + 1
    )
    xdawn_filter_i = sio.loadmat(xdawn_filter_i_dir)['filter']
    signal_train_i = xdawn_filter_i[np.newaxis, ...] @ signal_train_i

    # import mcmc output files
    mcmc_file_name = 'mcmc_seq_size_{}_cluster_xdawn_log_lhd_diff_approx_{}.mat'.format(seq_i + 1, log_lhd_approx)
    mcmc_output_dict_dir = '{}/{}/{}'.format(sub_new_trn_dir, cluster_name, channel_name)
    if hyper_param_bool:
        mcmc_output_dict_dir = '{}/{}'.format(mcmc_output_dict_dir, kernel_name)
    mcmc_output_dict_dir = '{}/{}'.format(mcmc_output_dict_dir, mcmc_file_name)
    mcmc_output_dict = sio.loadmat(mcmc_output_dict_dir, simplify_cells=True)

    chain_name_ls = list(mcmc_output_dict.keys())[3:]
    z_vector_mcmc = []
    for chain_name_iter in chain_name_ls:
        z_vector_mcmc.append(mcmc_output_dict[chain_name_iter]['z_vector'])
    z_vector_mcmc = np.concatenate(z_vector_mcmc, axis=0)
    z_vector_mean = np.mean(z_vector_mcmc, axis=0)
    print('z_vector_mean = {}'.format(z_vector_mean))
    z_vector_delta_binary = (z_vector_mean >= delta_threshold) + 0.0
    z_vector_delta_binary_sum = np.sum(z_vector_delta_binary)

    if not z_vector_delta_binary_sum:
        print('None of source participants are selected by threshold {}. '
              'Use BSM-Reference for prediction.'.format(delta_threshold))
        # update mcmc_output_dict with reference method first
        mcmc_output_dict_dir = '{}/{}/{}'.format(sub_new_trn_dir, ref_xdawn_name, channel_name)
        if hyper_param_bool:
            mcmc_output_dict_dir = '{}/{}'.format(mcmc_output_dict_dir, kernel_name)

        mcmc_file_name = 'mcmc_seq_size_{}_reference_xdawn.mat'.format(seq_i + 1)
        mcmc_output_dict_dir = '{}/{}'.format(mcmc_output_dict_dir, mcmc_file_name)
        mcmc_output_dict = sio.loadmat(mcmc_output_dict_dir)
        [mixture_prob_train_dict, _, mu_tar_mixture,
         mu_ntar_mixture, std_common_mixture] = predict_char_accuracy_multi(
            signal_train_i, label_train_i, code_train_i, mcmc_output_dict,
            seq_i + 1, xdawn_min, signal_length, eigen_fun_mat_dict['group_0'],
            target_char_size, 'ref'
        )
    else:
        print('At least one source participant is selected by threshold {}. '
              'Use BSM-Cluster for prediction.'.format(delta_threshold))
        [mixture_prob_train_dict, _, mu_tar_mixture,
         mu_ntar_mixture, std_common_mixture] = predict_char_accuracy_multi(
            signal_train_i, label_train_i, code_train_i, mcmc_output_dict,
            seq_i + 1, xdawn_min, signal_length, None,
            target_char_size, 'cluster'
        )

    mixture_predict_train_name_seq_i = 'mixture_xdawn_predict_train_seq_size_{}_z_threshold_{}'.format(
        seq_i + 1, delta_threshold
    )

    mixture_predict_train_name_dir_seq_i = '{}/{}/{}'.format(
        sub_new_trn_dir, mixture_xdawn_name, channel_name
    )
    if hyper_param_bool:
        mixture_predict_train_name_dir_seq_i = '{}/{}'.format(mixture_predict_train_name_dir_seq_i, kernel_name)
        if not os.path.exists(mixture_predict_train_name_dir_seq_i):
            os.mkdir(mixture_predict_train_name_dir_seq_i)
    mixture_predict_train_name_dir_seq_i = '{}/{}.json'.format(
        mixture_predict_train_name_dir_seq_i, mixture_predict_train_name_seq_i
    )
    with open(mixture_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(mixture_prob_train_dict, write_file)

    # testing set
    for test_dat_name_common in test_dat_name_common_ls:
        print('Testing data {}'.format(test_dat_name_common))
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

        if not z_vector_delta_binary_sum:
            print('None of source participants are selected. Use BSM-Reference for prediction.')
            [mixture_prob_test_dict, _, _, _, _] = predict_char_accuracy_multi(
                signal_test, None, code_test, mcmc_output_dict,
                seq_size_test, xdawn_min, signal_length, eigen_fun_mat_dict['group_0'],
                target_num_test, 'ref',
                mu_tar = mu_tar_mixture, mu_ntar = mu_ntar_mixture, std_common = std_common_mixture
            )
        else:
            print('At least one source participant is selected by threshold {}. '
                  'Use BSM-Cluster for prediction.'.format(delta_threshold))
            [mixture_prob_test_dict, _, _, _, _] = predict_char_accuracy_multi(
                signal_test, None, code_test, mcmc_output_dict,
                seq_size_test, xdawn_min, signal_length, None,
                target_num_test, 'cluster',
                mu_tar=mu_tar_mixture, mu_ntar=mu_ntar_mixture, std_common=std_common_mixture
            )
        mixture_predict_test_name_seq_i = 'mixture_xdawn_predict_test_seq_size_{}_z_threshold_{}'.format(seq_i + 1, delta_threshold)

        mixture_predict_test_name_dir_seq_i = '{}/{}/{}/{}'.format(
            sub_new_test_dir, test_dat_name_common, mixture_xdawn_name, channel_name
        )
        if hyper_param_bool:
            mixture_predict_test_name_dir_seq_i = '{}/{}'.format(mixture_predict_test_name_dir_seq_i, kernel_name)
        mixture_predict_test_name_dir_seq_i = '{}/{}.json'.format(
            mixture_predict_test_name_dir_seq_i, mixture_predict_test_name_seq_i
        )
        with open(mixture_predict_test_name_dir_seq_i, "w") as write_file:
            json.dump(mixture_prob_test_dict, write_file)

