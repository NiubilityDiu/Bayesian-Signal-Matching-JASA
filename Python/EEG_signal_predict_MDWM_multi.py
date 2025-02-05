from self_py_fun.EEGGlobal import *
from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGFun import *
from self_py_fun.ExistMLFun import *
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)
signal_length = 25
tol = 1e-3

if local_use:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_local)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_local)
    sub_new_name = 'K143'
    seq_source = 5
    reduced_bool = True

else:
    parent_trn_dir = '{}/TRN_files'.format(parent_path_slurm)
    parent_frt_dir = '{}/FRT_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_source = int(sys.argv[2])
    reduced_bool = (sys.argv[3] == 'T' or sys.argv[3] == 'True')

seq_size_train = 15
if sub_new_name in ['K154', 'K190']:
    seq_size_train = 20

source_sub_name_ls = np.copy(sub_name_ls).tolist()
source_sub_name_ls.remove(sub_new_name)

# use smaller source pool
if reduced_bool:
    for ele in sorted(sub_throw_name_ls, reverse=True):
        if ele in source_sub_name_ls:
            source_sub_name_ls.remove(ele)
    sub_name_reduce_ls = np.copy(np.array(source_sub_name_ls)).tolist()
    sub_name_reduce_ls.insert(0, sub_new_name)
    # update N_total here
    N_total = len(sub_name_reduce_ls)
else:
    sub_name_reduce_ls = np.copy(sub_name_ls).tolist()
    sub_name_reduce_ls.remove(sub_new_name)
    sub_name_reduce_ls.insert(0, sub_new_name)

sub_new_trn_dir = '{}/{}'.format(parent_trn_dir, sub_new_name)
trn_dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
sub_new_trn_dat_dir = '{}/{}_{}.mat'.format(
    sub_new_trn_dir, sub_new_name, trn_dat_name_common
)

sub_new_trn_mdwm_dir = '{}/MDWM'.format(sub_new_trn_dir)
if not os.path.exists('{}'.format(sub_new_trn_mdwm_dir)):
    os.mkdir(sub_new_trn_mdwm_dir)
sub_new_trn_mdwm_dir = '{}/channel_{}'.format(sub_new_trn_mdwm_dir, select_channel_ids_str)
if not os.path.exists('{}'.format(sub_new_trn_mdwm_dir)):
    os.mkdir(sub_new_trn_mdwm_dir)

sub_new_test_dir = '{}/{}'.format(parent_frt_dir, sub_new_name)
test_dat_name_common_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
if sub_new_name in FRT_file_name_dict.keys():
    test_dat_name_common_ls = FRT_file_name_dict[sub_new_name]

# create new folder (e.g., FRT_files/K106/001_BCI_FRT/N_41_K_5_gibbs/channel_6/ (seq_i)...)
for test_dat_name_common in test_dat_name_common_ls:
    sub_new_test_dir_2 = '{}/{}'.format(sub_new_test_dir, test_dat_name_common)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_2 = '{}/MDWM'.format(sub_new_test_dir_2)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)
    sub_new_test_dir_2 = '{}/channel_{}'.format(sub_new_test_dir_2, select_channel_ids_str)
    if not os.path.exists('{}'.format(sub_new_test_dir_2)):
        os.mkdir(sub_new_test_dir_2)

# seq_i = seq_source - 1
for seq_i in np.array([seq_source-1]):
    print('source sequence size {}'.format(seq_i+1))

    # import eeg_data
    source_data, new_data = import_eeg_data_cluster_format(
        sub_name_reduce_ls, sub_new_name, target_char_size,
        select_channel_ids, signal_length,
        letter_dim_sub, seq_i, seq_source, parent_trn_dir, trn_dat_name_common,
        reshape_2d_bool=False, xdawn_bool=False, n_components=16,
        lower_seq_id=1, pseudo_bool=False
    )

    domain_tradeoff = 0.5
    mdwm_obj, mdwm_ERP_cov_obj, X_domain_mdwm_size, mdwm_param_dict = mdwm_fit_from_eeg_feature(
        new_data, source_data, source_sub_name_ls, domain_tradeoff, tol
    )

    ############################
    ## Prediction on TRN file ##
    ############################
    # mu_tar_mdwm = mdwm_param_dict['mu_tar']
    # mu_ntar_mdwm = mdwm_param_dict['mu_ntar']
    # std_common_mdwm = mdwm_param_dict['std_common']

    # import training set (up to seq_i+1)
    signal_train_i, label_train_i, code_train_i = import_eeg_train_data(
        sub_new_trn_dat_dir, seq_i, select_channel_ids, E_total, signal_length,
        target_num_train, seq_size_train, reshape_2d_bool=False
    )
    # signal_train_cov = mdwm_ERP_cov_obj.transform(signal_train_i) + np.eye(X_domain_mdwm_size) * tol
    # pred_score_train_0 = mdwm_obj.predict_log_proba(signal_train_cov)[:, 0]
    # pred_binary_train_0 = mdwm_obj.predict(signal_train_cov)
    # pred_prob_letter_train, pred_prob_mat_train = ml_predict_letter_likelihood(
    #     pred_score_train_0, code_train_i, target_num_train,
    #     seq_i+1, mu_tar_mdwm, mu_ntar_mdwm, std_common_mdwm,
    #     stimulus_group_set, sim_rcp_array
    # )
    # mdwm_prob_train_dict = {
    #     'letter': pred_prob_letter_train.tolist(),
    #     'prob': pred_prob_mat_train.tolist()
    # }

    [mdwm_prob_train_dict, mdwm_train_obj, mu_tar_mdwm,
     mu_ntar_mdwm, std_common_mdwm] = predict_char_accuracy_multi(
            signal_train_i, label_train_i, code_train_i, None,
            seq_i + 1, None, signal_length, None,
            target_char_size, 'mdwm',
            domain_tradeoff=domain_tradeoff, tol=tol,
            new_data=new_data, source_data=source_data,
            source_sub_name=source_sub_name_ls,
            mdwm_obj=None, X_domain_mdwm_size=None
        )
    mdwm_predict_train_name_seq_i = 'MDWM_predict_train_seq_size_{}'.format(seq_i + 1)
    mdwm_predict_train_name_dir_seq_i = '{}/MDWM/channel_{}/{}.json'.format(
        sub_new_trn_dir, select_channel_ids_str, mdwm_predict_train_name_seq_i
    )
    with open(mdwm_predict_train_name_dir_seq_i, "w") as write_file:
        json.dump(mdwm_prob_train_dict, write_file)

    mdwm_train_fit_obj = mdwm_train_obj['mdwm_obj']
    mdwm_train_ERP_cov_obj = mdwm_train_obj['mdwm_ERP_cov_obj']

    # testing set
    for test_dat_name_common in test_dat_name_common_ls:
        print(test_dat_name_common)

        test_data_dir = '{}/{}_{}_Truncated_Data_0.5_6.mat'.format(
            sub_new_test_dir, sub_new_name, test_dat_name_common
        )
        signal_test_dict = import_eeg_test_data(
            test_data_dir, select_channel_ids, E_total, signal_length,
            reshape_2d_bool=False
        )
        signal_test = signal_test_dict['Signal']
        code_test = signal_test_dict['Code']
        seq_size_test = signal_test_dict['Seq_size']
        target_num_test = signal_test_dict['Text_size']

        # signal_test_cov = mdwm_ERP_cov_obj.transform(signal_test)+np.eye(X_domain_mdwm_size) * tol
        # pred_score_test_0 = mdwm_obj.predict_log_proba(signal_test_cov)[:, 0]
        # pred_binary_test_0 = mdwm_obj.predict(signal_test_cov)
        #
        # pred_prob_letter_test, pred_prob_mat_test = ml_predict_letter_likelihood(
        #     pred_score_test_0, code_test, target_num_test,
        #     seq_size_test, mu_tar_mdwm, mu_ntar_mdwm, std_common_mdwm,
        #     stimulus_group_set, sim_rcp_array
        # )
        # mdwm_prob_test_dict = {
        #     'letter': pred_prob_letter_test.tolist(),
        #     'prob': pred_prob_mat_test.tolist()
        # }
        [mdwm_prob_test_dict, _, _, _, _] = predict_char_accuracy_multi(
            signal_test, None, code_test, None,
            seq_size_test, None, signal_length, None,
            target_num_test, 'mdwm',
            domain_tradeoff=domain_tradeoff, tol=tol,
            mdwm_obj=mdwm_train_fit_obj,
            mdwm_ERP_cov_obj=mdwm_train_ERP_cov_obj,
            X_domain_mdwm_size=X_domain_mdwm_size,
            mu_tar=mu_tar_mdwm, mu_ntar=mu_ntar_mdwm,
            std_common=std_common_mdwm,
        )
        mdwm_predict_test_name_seq_i = 'MDWM_predict_test_seq_size_{}'.format(seq_i + 1)
        mdwm_predict_test_name_dir_seq_i = '{}/{}/MDWM/channel_{}/{}.json'.format(
            sub_new_test_dir, test_dat_name_common, select_channel_ids_str, mdwm_predict_test_name_seq_i
        )

        with open(mdwm_predict_test_name_dir_seq_i, "w") as write_file:
            json.dump(mdwm_prob_test_dict, write_file)
    

