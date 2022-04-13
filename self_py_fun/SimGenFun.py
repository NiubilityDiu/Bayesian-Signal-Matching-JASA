from self_py_fun.SimFun import *


def generate_simulated_data_score_based(
        param_true_dict, rcp_array, scenario_name_dir, seed_num=612
):

    target_char = param_true_dict['target_char']
    q_mat, _ = compute_q_matrix(target_char, rcp_array)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']

    w_order = np.arange(rcp_unit_flash_num) + 1
    np.random.seed(seed_num)

    sim_data_dict = {}

    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_dict[subject_n_name] = {}
        group_label_n = mtn.rvs(n=1, p=param_true_dict['label_N'][subject_n_name], size=1)
        group_label_n = np.squeeze(np.where(np.squeeze(group_label_n, axis=0) == 1)[0])
        # print(group_label_n)
        group_label_name = 'group_{}'.format(group_label_n)
        mu_n = param_true_dict[group_label_name]['mu']
        sigma_n = param_true_dict[group_label_name]['sigma']
        seq_n = seq_n_ls[n]
        sim_data_dict[subject_n_name]['label'] = group_label_n

        w_mat = np.zeros([seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([seq_n, rcp_unit_flash_num])

        for seq_i in range(seq_n):
            # np.random.shuffle(w_order)
            w_mat[seq_i, :] = np.copy(w_order)
            p_mat_i = compute_p_matrix(w_order)
            t_mat_i = np.matmul(p_mat_i, q_mat)
            mu_i = np.squeeze(np.matmul(t_mat_i, mu_n), axis=-1)
            x_mat[seq_i, :] = mvn.rvs(mean=mu_i, cov=sigma_n**2 * np.eye(rcp_unit_flash_num), size=1)

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat)
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat)

    sio.savemat('{}/sim_dat.mat'.format(scenario_name_dir), sim_data_dict)

    return sim_data_dict


def generate_simulated_data_signal_based(
        param_true_dict, rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char = param_true_dict['target_char']
    q_mat, _ = compute_q_matrix(target_char, rcp_array)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']
    signal_length = param_true_dict['signal_length']

    w_order = np.arange(rcp_unit_flash_num) + 1

    sim_data_dict = {'omega': target_char}
    target_row_num, target_col_num = determine_row_column_indices(target_char)

    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_dict[subject_n_name] = {}
        group_label_n = mtn.rvs(n=1, p=param_true_dict['label_N'][subject_n_name], size=1)
        group_label_n = np.squeeze(np.where(np.squeeze(group_label_n, axis=0) == 1)[0])

        group_label_name = 'group_{}'.format(group_label_n)
        beta_tar_n = param_true_dict[group_label_name]['beta_tar']
        beta_ntar_n = param_true_dict[group_label_name]['beta_ntar']
        beta_n = np.stack([beta_tar_n, beta_ntar_n], axis=0)
        # print('beta_n has shape {}'.format(beta_n.shape))  # (2, 25)
        sigma_n = param_true_dict[group_label_name]['sigma']

        if 'rho' in param_true_dict[group_label_name].keys():
            rho_n = param_true_dict[group_label_name]['rho']
            cov_mat_n = create_exponential_decay_cov_mat(sigma_n**2, rho_n, signal_length)
        else:
            cov_mat_n = sigma_n**2 * np.eye(signal_length)  # start from identity matrix

        seq_n = seq_n_ls[n]
        sim_data_dict[subject_n_name]['label'] = group_label_n.tolist()

        w_mat = np.zeros([seq_n, rcp_unit_flash_num])
        y_mat = np.zeros([seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([seq_n, rcp_unit_flash_num, signal_length])

        for seq_i in range(seq_n):
            # np.random.shuffle(w_order)
            w_mat[seq_i, :] = np.copy(w_order)
            y_mat[seq_i, np.where(w_order == target_row_num)[0][0]] = 1
            y_mat[seq_i, np.where(w_order == target_col_num)[0][0]] = 1
            p_mat_i = compute_p_matrix(w_order)
            t_mat_i = np.matmul(p_mat_i, q_mat)
            mu_i = np.matmul(t_mat_i, beta_n)  # reorder mu matrix by W_{n,i} and Y
            # print('mu_i has shape {}'.format(mu_i.shape))  # (12, 25)
            for flash_j in range(rcp_unit_flash_num):
                x_mat[seq_i, flash_j, :] = mvn.rvs(
                    mean=mu_i[flash_j, :], cov=cov_mat_n, size=1
                )

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat).tolist()
        sim_data_dict[subject_n_name]['Y'] = np.copy(y_mat).tolist()
        # x_mat = np.reshape(x_mat, [seq_n, rcp_unit_flash_num * signal_length])
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_test_data_signal_based(
        param_true_dict, new_subject_label: int, target_char: list, seq_size_new,
        rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char_len = len(target_char)
    signal_length = param_true_dict['signal_length']

    sim_data_dict = {'omega': target_char}
    # modify here to allow subject 0 to have different cluster assignment.
    new_subject_label_name = 'group_{}'.format(new_subject_label)
    beta_tar_new = param_true_dict[new_subject_label_name]['beta_tar']
    beta_ntar_new = param_true_dict[new_subject_label_name]['beta_ntar']
    beta_new = np.stack([beta_tar_new, beta_ntar_new], axis=0)
    sigma_new = param_true_dict[new_subject_label_name]['sigma']
    if 'rho' in param_true_dict[new_subject_label_name].keys():
        rho_new = param_true_dict[new_subject_label_name]['rho']
        cov_mat_new = create_exponential_decay_cov_mat(sigma_new ** 2, rho_new, signal_length)
    else:
        cov_mat_new = sigma_new ** 2 * np.eye(signal_length)  # start from identity matrix

    w_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    y_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    x_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num, signal_length])

    w_order = np.arange(rcp_unit_flash_num) + 1

    for char_iter in range(target_char_len):
        target_char_iter = target_char[char_iter]
        char_iter_row, char_iter_col = determine_row_column_indices(target_char_iter)
        q_mat_char, _ = compute_q_matrix(target_char_iter, rcp_array)
        for seq_i in range(seq_size_new):
            np.random.shuffle(w_order)
            w_mat[char_iter, seq_i, :] = np.copy(w_order)
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_row)[0][0]] = 1
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_col)[0][0]] = 1
            p_mat_iter = compute_p_matrix(w_order)
            t_mat_iter = np.matmul(p_mat_iter, q_mat_char)
            mu_iter = np.matmul(t_mat_iter, beta_new)  # reorder mu matrix by W_{n,i} and Y
            # print('mu_i has shape {}'.format(mu_iter.shape))
            for flash_j in range(rcp_unit_flash_num):
                x_mat[char_iter, seq_i, flash_j, :] = mvn.rvs(
                    mean=mu_iter[flash_j, :], cov=cov_mat_new, size=1
                )

    sim_data_dict['W'] = np.copy(w_mat).tolist()
    sim_data_dict['Y'] = np.copy(y_mat).tolist()
    sim_data_dict['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat_test.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_data_multi_channel(
        param_true_dict, rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char = param_true_dict['target_char']
    q_mat, _ = compute_q_matrix(target_char, rcp_array)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']
    signal_length = param_true_dict['signal_length']
    channel_dim = param_true_dict['channel_dim']

    w_order = np.arange(rcp_unit_flash_num) + 1

    sim_data_dict = {'omega': target_char}
    target_row_num, target_col_num = determine_row_column_indices(target_char)

    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_dict[subject_n_name] = {}
        group_label_n = mtn.rvs(n=1, p=param_true_dict['label_N'][subject_n_name], size=1)
        group_label_n = np.squeeze(np.where(np.squeeze(group_label_n, axis=0) == 1)[0])

        group_label_name = 'group_{}'.format(group_label_n)
        beta_tar_n = param_true_dict[group_label_name]['beta_tar']  # (e, 25)
        beta_ntar_n = param_true_dict[group_label_name]['beta_ntar']

        rho_n = param_true_dict[group_label_name]['rho']
        cov_mat_rho_n = create_exponential_decay_cov_mat(1.0, rho_n, signal_length)

        lambda_n = param_true_dict[group_label_name]['lambda']
        sigma_sq_n = param_true_dict[group_label_name]['sigma_sq']  # (e,)
        cov_mat_lambda_n = create_cs_cov_mat(sigma_sq_n, lambda_n, channel_dim)

        seq_n = seq_n_ls[n]
        sim_data_dict[subject_n_name]['label'] = group_label_n.tolist()

        w_mat = np.zeros([seq_n, rcp_unit_flash_num])
        y_mat = np.zeros([seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([seq_n, rcp_unit_flash_num, channel_dim, signal_length])

        for seq_i in range(seq_n):
            np.random.shuffle(w_order)
            w_mat[seq_i, :] = np.copy(w_order)
            y_mat[seq_i, np.where(w_order == target_row_num)[0][0]] = 1
            y_mat[seq_i, np.where(w_order == target_col_num)[0][0]] = 1
            for flash_j in range(rcp_unit_flash_num):
                if y_mat[seq_i, flash_j] == 1:
                    mean_mat_ij = beta_tar_n
                else:
                    mean_mat_ij = beta_ntar_n
                x_mat[seq_i, flash_j, ...] = stats.matrix_normal(
                    mean=mean_mat_ij, rowcov=cov_mat_lambda_n, colcov=cov_mat_rho_n
                ).rvs(size=1)

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat).tolist()
        sim_data_dict[subject_n_name]['Y'] = np.copy(y_mat).tolist()
        # x_mat = np.reshape(x_mat, [seq_n, rcp_unit_flash_num * signal_length])
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_test_data_multi_channel(
        param_true_dict, new_subject_label: int, target_char: list, seq_size_new,
        rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char_len = len(target_char)
    signal_length = param_true_dict['signal_length']
    channel_dim = param_true_dict['channel_dim']

    sim_data_dict = {'omega': target_char}
    # modify here to allow subject 0 to have different cluster assignment.
    new_subject_label_name = 'group_{}'.format(new_subject_label)
    beta_tar_new = param_true_dict[new_subject_label_name]['beta_tar']
    beta_ntar_new = param_true_dict[new_subject_label_name]['beta_ntar']

    rho_new = param_true_dict[new_subject_label_name]['rho']
    cov_mat_rho_new = create_exponential_decay_cov_mat(1.0, rho_new, signal_length)

    lambda_new = param_true_dict[new_subject_label_name]['lambda']
    sigma_sq_new = param_true_dict[new_subject_label_name]['sigma_sq']
    cov_mat_lambda_new = create_cs_cov_mat(sigma_sq_new, lambda_new, channel_dim)

    w_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    y_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    x_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num, channel_dim, signal_length])

    w_order = np.arange(rcp_unit_flash_num) + 1

    for char_iter in range(target_char_len):
        target_char_iter = target_char[char_iter]
        char_iter_row, char_iter_col = determine_row_column_indices(target_char_iter)
        q_mat_char, _ = compute_q_matrix(target_char_iter, rcp_array)

        for seq_i in range(seq_size_new):
            np.random.shuffle(w_order)
            w_mat[char_iter, seq_i, :] = np.copy(w_order)
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_row)[0][0]] = 1
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_col)[0][0]] = 1

            for flash_j in range(rcp_unit_flash_num):
                '''
                # 1. without using matrix normal distribution
                err_ij_spatial = mvn.rvs(mean=np.zeros([channel_dim]), cov=cov_mat_lambda_new, size=1)[:, np.newaxis]
                err_ij_temporal = mvn.rvs(mean=np.zeros([signal_length]), cov=cov_mat_rho_new, size=1)[np.newaxis, :]
                err_ij = err_ij_spatial + err_ij_temporal
                if y_mat[char_iter, seq_i, flash_j] == 1:
                    x_mat[char_iter, seq_i, flash_j, ...] = beta_tar_new + err_ij
                else:
                    x_mat[char_iter, seq_i, flash_j, ...] = beta_ntar_new + err_ij
                '''
                # 2. using matrix normal distribution or vectorization reparametrization
                if y_mat[char_iter, seq_i, flash_j] == 1:
                    mean_mat_ij = beta_tar_new
                else:
                    mean_mat_ij = beta_ntar_new
                x_mat[char_iter, seq_i, flash_j, ...] = stats.matrix_normal(
                    mean=mean_mat_ij, rowcov=cov_mat_lambda_new, colcov=cov_mat_rho_new
                ).rvs(size=1)

    sim_data_dict['W'] = np.copy(w_mat).tolist()
    sim_data_dict['Y'] = np.copy(y_mat).tolist()
    sim_data_dict['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat_test.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict
