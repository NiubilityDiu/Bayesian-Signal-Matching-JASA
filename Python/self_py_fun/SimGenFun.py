from self_py_fun.SimFun import *
import scipy.io as sio
from scipy.stats import multinomial as mtn
from scipy.stats import matrix_normal as matn
import matplotlib.pyplot as plt
plt.style.use("bmh")


def generate_simulated_data(
        param_true_dict, rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char = param_true_dict['target_char']
    target_char_len = len(target_char)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']
    signal_length = param_true_dict['signal_length']

    w_order = np.arange(rcp_unit_flash_num) + 1
    sim_data_dict = {'omega': target_char}

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

        w_mat = np.zeros([target_char_len, seq_n, rcp_unit_flash_num])
        y_mat = np.zeros([target_char_len, seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([target_char_len, seq_n, rcp_unit_flash_num, signal_length])

        for char_iter in range(target_char_len):
            w_mat_iter, y_mat_iter, x_mat_iter = _generate_simulated_data_inside(beta_n, cov_mat_n,
                                                                                 target_char[char_iter], w_order, seq_n,
                                                                                 signal_length, rcp_array, False)
            w_mat[char_iter, ...] = w_mat_iter
            y_mat[char_iter, ...] = y_mat_iter
            x_mat[char_iter, ...] = x_mat_iter

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat).tolist()
        sim_data_dict[subject_n_name]['Y'] = np.copy(y_mat).tolist()
        # x_mat = np.reshape(x_mat, [seq_n, rcp_unit_flash_num * signal_length])
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_test_data(
        param_true_dict, new_subject_label: int, target_char: list, seq_size_new,
        rcp_array, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char_len = len(target_char)
    signal_length = param_true_dict['signal_length']

    sim_data_test_dict = {'omega': target_char}
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

    w_mat_new = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    y_mat_new = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    x_mat_new = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num, signal_length])

    w_order = np.arange(rcp_unit_flash_num) + 1

    for char_iter in range(target_char_len):
        target_char_iter = target_char[char_iter]
        w_mat_new_iter, y_mat_new_iter, x_mat_new_iter = _generate_simulated_data_inside(beta_new, cov_mat_new,
                                                                                         target_char_iter, w_order,
                                                                                         seq_size_new, signal_length,
                                                                                         rcp_array, True)
        w_mat_new[char_iter, ...] = w_mat_new_iter
        y_mat_new[char_iter, ...] = y_mat_new_iter
        x_mat_new[char_iter, ...] = x_mat_new_iter

    sim_data_test_dict['W'] = np.copy(w_mat_new).tolist()
    sim_data_test_dict['Y'] = np.copy(y_mat_new).tolist()
    sim_data_test_dict['X'] = np.copy(x_mat_new).tolist()

    sim_data_test_dict_dir = '{}/sim_dat_test.json'.format(scenario_name_dir)
    with open(sim_data_test_dict_dir, "w") as write_file:
        json.dump(sim_data_test_dict, write_file)

    return sim_data_test_dict


def _generate_simulated_data_inside(
        beta_vec, cov_mat,
        target_char, w_order, seq_size, signal_length, rcp_array,
        shuffle_bool=True
):
    char_row, char_col = determine_row_column_indices(target_char)
    q_mat_char, _ = compute_q_matrix(target_char, rcp_array)
    w_mat = np.zeros([seq_size, rcp_unit_flash_num])
    y_mat = np.zeros([seq_size, rcp_unit_flash_num])
    x_mat = np.zeros([seq_size, rcp_unit_flash_num, signal_length])

    for seq_i in range(seq_size):
        if shuffle_bool:
            np.random.shuffle(w_order)

        w_mat[seq_i, :] = np.copy(w_order)
        y_mat[seq_i, np.where(w_order == char_row)[0][0]] = 1
        y_mat[seq_i, np.where(w_order == char_col)[0][0]] = 1
        p_mat_iter = compute_p_matrix(w_order)
        t_mat_iter = np.matmul(p_mat_iter, q_mat_char)
        mu_iter = np.matmul(t_mat_iter, beta_vec)  # reorder mu matrix by W_{n,i} and Y
        # print('mu_i has shape {}'.format(mu_iter.shape))
        for flash_j in range(rcp_unit_flash_num):
            x_mat[seq_i, flash_j, :] = mvn.rvs(
                mean=mu_iter[flash_j, :], cov=cov_mat, size=1
            )
    return w_mat, y_mat, x_mat


def import_mcmc_source_mat(source_mcmc_dir):
    mcmc_mat = sio.loadmat(source_mcmc_dir)
    beta_tar_mean = np.mean(mcmc_mat['beta_tar'], axis=(0, 3))
    beta_ntar_mean = np.mean(mcmc_mat['beta_ntar'], axis=(0, 3))

    # plt.plot(beta_tar_mean[e_id - 1, :], color='red', label='target')
    # plt.plot(beta_ntar_mean[e_id - 1, :], color='blue', label='non-target')
    # plt.title(group_name)
    # plt.legend(loc='best')
    # plt.show()

    return beta_tar_mean, beta_ntar_mean


def generate_simulated_data_multi(
        param_true_dict, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char = param_true_dict['target_char']
    # q_mat, _ = compute_q_matrix(target_char, rcp_array)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']
    signal_length = param_true_dict['signal_length']
    E = param_true_dict['E']
    w_order = np.arange(rcp_unit_flash_num) + 1

    sim_data_dict = {'omega': target_char}
    target_char_size = len(target_char)

    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_dict[subject_n_name] = {}

        group_label_n = mtn.rvs(n=1, p=param_true_dict['label_N'][subject_n_name], size=1)
        group_label_n = np.squeeze(np.where(np.squeeze(group_label_n, axis=0) == 1)[0])

        group_label_name = 'group_{}'.format(group_label_n)
        beta_tar_n = param_true_dict[group_label_name]['beta_tar']  # (E, signal_length)
        beta_ntar_n = param_true_dict[group_label_name]['beta_ntar']  # (E, signal_length)

        rho_n = param_true_dict[group_label_name]['rho']
        cov_mat_temp_n = create_exponential_decay_cov_mat(1.0, rho_n, signal_length)

        eta_n = param_true_dict[group_label_name]['eta']
        sigma_n = param_true_dict[group_label_name]['sigma']  # (E,)
        cov_mat_spatial_n = create_cs_cov_mat(sigma_n ** 2, eta_n, E)

        matn_n_tar_obj = matn(mean=beta_tar_n, rowcov=cov_mat_spatial_n, colcov=cov_mat_temp_n)
        matn_n_ntar_obj = matn(mean=beta_ntar_n, rowcov=cov_mat_spatial_n, colcov=cov_mat_temp_n)

        seq_n = seq_n_ls[n]
        sim_data_dict[subject_n_name]['label'] = group_label_n.tolist()

        w_mat = np.zeros([target_char_size, seq_n, rcp_unit_flash_num])
        y_mat = np.zeros([target_char_size, seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([target_char_size, seq_n, rcp_unit_flash_num, E, signal_length])

        for char_i in range(target_char_size):
            target_char_iter = target_char[char_i]
            target_char_i_row, target_char_i_col = determine_row_column_indices(target_char_iter)

            for seq_i in range(seq_n):
                np.random.shuffle(w_order)
                w_mat[char_i, seq_i, :] = np.copy(w_order)
                y_mat[char_i, seq_i, np.where(w_order == target_char_i_row)[0][0]] = 1
                y_mat[char_i, seq_i, np.where(w_order == target_char_i_col)[0][0]] = 1
                for flash_j in range(rcp_unit_flash_num):
                    if y_mat[char_i, seq_i, flash_j] == 1:
                        x_mat[char_i, seq_i, flash_j, ...] = matn_n_tar_obj.rvs(size=1)
                    else:
                        x_mat[char_i, seq_i, flash_j, ...] = matn_n_ntar_obj.rvs(size=1)

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat).tolist()
        sim_data_dict[subject_n_name]['Y'] = np.copy(y_mat).tolist()
        # x_mat = np.reshape(x_mat, [seq_n, rcp_unit_flash_num * signal_length])
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_test_data_multi(
        param_true_dict, new_subject_label: int, target_char: list, seq_size_new,
        scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char_len = len(target_char)
    signal_length = param_true_dict['signal_length']
    E = param_true_dict['E']

    sim_data_dict = {'omega': target_char}
    # modify here to allow subject 0 to have different cluster assignment.
    new_subject_label_name = 'group_{}'.format(new_subject_label)
    beta_tar_new = param_true_dict[new_subject_label_name]['beta_tar']
    beta_ntar_new = param_true_dict[new_subject_label_name]['beta_ntar']

    rho_new = param_true_dict[new_subject_label_name]['rho']
    cov_mat_temp_new = create_exponential_decay_cov_mat(1.0, rho_new, signal_length)

    eta_new = param_true_dict[new_subject_label_name]['eta']
    sigma_new = param_true_dict[new_subject_label_name]['sigma']
    cov_mat_spatial_new = create_cs_cov_mat(sigma_new ** 2, eta_new, E)

    matn_new_tar_obj = matn(mean=beta_tar_new, rowcov=cov_mat_spatial_new, colcov=cov_mat_temp_new)
    matn_new_ntar_obj = matn(mean=beta_ntar_new, rowcov=cov_mat_spatial_new, colcov=cov_mat_temp_new)

    w_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    y_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    x_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num, E, signal_length])

    w_order = np.arange(rcp_unit_flash_num) + 1

    for char_iter in range(target_char_len):
        target_char_iter = target_char[char_iter]
        char_iter_row, char_iter_col = determine_row_column_indices(target_char_iter)
        # q_mat_char, _ = compute_q_matrix(target_char_iter, rcp_array)

        for seq_i in range(seq_size_new):
            np.random.shuffle(w_order)
            w_mat[char_iter, seq_i, :] = np.copy(w_order)
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_row)[0][0]] = 1
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_col)[0][0]] = 1

            for flash_j in range(rcp_unit_flash_num):
                if y_mat[char_iter, seq_i, flash_j] == 1:
                    x_mat[char_iter, seq_i, flash_j, ...] = matn_new_tar_obj.rvs(size=1)
                else:
                    x_mat[char_iter, seq_i, flash_j, ...] = matn_new_ntar_obj.rvs(size=1)

    sim_data_dict['W'] = np.copy(w_mat).tolist()
    sim_data_dict['Y'] = np.copy(y_mat).tolist()
    sim_data_dict['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat_test.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_data_multi_real(
        param_true_dict, noise_dict, scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char = param_true_dict['target_char']
    # q_mat, _ = compute_q_matrix(target_char, rcp_array)

    N = param_true_dict['N']
    # K = param_true_dict['K']
    seq_n_ls = param_true_dict['seq_size']
    signal_length = param_true_dict['signal_length']
    E = param_true_dict['E']

    w_order = np.arange(rcp_unit_flash_num) + 1

    sim_data_dict = {'omega': target_char}
    target_row_num, target_col_num = determine_row_column_indices(target_char)

    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_dict[subject_n_name] = {}
        group_label_n = mtn.rvs(n=1, p=param_true_dict['label_N'][subject_n_name], size=1)
        group_label_n = np.squeeze(np.where(np.squeeze(group_label_n, axis=0) == 1)[0])

        group_label_name = 'group_{}'.format(group_label_n)
        beta_tar_n = param_true_dict[group_label_name]['beta_tar']  # (E, signal_length)
        beta_ntar_n = param_true_dict[group_label_name]['beta_ntar'] # (E, signal_length)

        noise_tar_n_group = noise_dict[group_label_name]['target']
        noise_tar_n_size = noise_dict[group_label_name]['target'].shape[0]

        noise_ntar_n_group = noise_dict[group_label_name]['non-target']
        noise_ntar_n_size = noise_dict[group_label_name]['non-target'].shape[0]

        seq_n = seq_n_ls[n]
        sim_data_dict[subject_n_name]['label'] = group_label_n.tolist()

        w_mat = np.zeros([seq_n, rcp_unit_flash_num])
        y_mat = np.zeros([seq_n, rcp_unit_flash_num])
        x_mat = np.zeros([seq_n, rcp_unit_flash_num, E, signal_length])

        for seq_i in range(seq_n):
            np.random.shuffle(w_order)
            w_mat[seq_i, :] = np.copy(w_order)
            y_mat[seq_i, np.where(w_order == target_row_num)[0][0]] = 1
            y_mat[seq_i, np.where(w_order == target_col_num)[0][0]] = 1
            for flash_j in range(rcp_unit_flash_num):
                if y_mat[seq_i, flash_j] == 1:
                    random_tar_i_j = np.random.random_integers(low=0, high=noise_tar_n_size-1, size=None)
                    x_mat[seq_i, flash_j, ...] = beta_tar_n + noise_tar_n_group[random_tar_i_j, ...]
                else:
                    random_ntar_i_j = np.random.random_integers(low=0, high=noise_ntar_n_size-1, size=None)
                    x_mat[seq_i, flash_j, ...] = beta_ntar_n + noise_ntar_n_group[random_ntar_i_j, ...]

        sim_data_dict[subject_n_name]['W'] = np.copy(w_mat).tolist()
        sim_data_dict[subject_n_name]['Y'] = np.copy(y_mat).tolist()
        # x_mat = np.reshape(x_mat, [seq_n, rcp_unit_flash_num * signal_length])
        sim_data_dict[subject_n_name]['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict


def generate_simulated_test_data_multi_real(
        param_true_dict, noise_dict: dict, new_subject_label: int, target_char: list, seq_size_new,
        scenario_name_dir, seed_num=0
):
    np.random.seed(seed_num)
    target_char_len = len(target_char)
    signal_length = param_true_dict['signal_length']
    E = param_true_dict['E']

    sim_data_dict = {'omega': target_char}
    # modify here to allow subject 0 to have different cluster assignment.
    new_subject_label_name = 'group_{}'.format(new_subject_label)
    beta_tar_new = param_true_dict[new_subject_label_name]['beta_tar']
    beta_ntar_new = param_true_dict[new_subject_label_name]['beta_ntar']

    noise_tar_n_group = noise_dict[new_subject_label_name]['target']
    noise_tar_n_size = noise_dict[new_subject_label_name]['target'].shape[0]

    noise_ntar_n_group = noise_dict[new_subject_label_name]['non-target']
    noise_ntar_n_size = noise_dict[new_subject_label_name]['non-target'].shape[0]

    w_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    y_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num])
    x_mat = np.zeros([target_char_len, seq_size_new, rcp_unit_flash_num, E, signal_length])

    w_order = np.arange(rcp_unit_flash_num) + 1

    for char_iter in range(target_char_len):
        target_char_iter = target_char[char_iter]
        char_iter_row, char_iter_col = determine_row_column_indices(target_char_iter)
        # q_mat_char, _ = compute_q_matrix(target_char_iter, rcp_array)

        for seq_i in range(seq_size_new):
            np.random.shuffle(w_order)
            w_mat[char_iter, seq_i, :] = np.copy(w_order)
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_row)[0][0]] = 1
            y_mat[char_iter, seq_i, np.where(w_order == char_iter_col)[0][0]] = 1

            for flash_j in range(rcp_unit_flash_num):
                if y_mat[char_iter, seq_i, flash_j] == 1:
                    random_tar_i_j = np.random.random_integers(low=0, high=noise_tar_n_size-1, size=None)
                    x_mat[char_iter, seq_i, flash_j, ...] = beta_tar_new + noise_tar_n_group[random_tar_i_j, ...]
                else:
                    random_ntar_i_j = np.random.random_integers(low=0, high=noise_ntar_n_size-1, size=None)
                    x_mat[char_iter, seq_i, flash_j, ...] = beta_ntar_new + noise_ntar_n_group[random_ntar_i_j, ...]
    sim_data_dict['W'] = np.copy(w_mat).tolist()
    sim_data_dict['Y'] = np.copy(y_mat).tolist()
    sim_data_dict['X'] = np.copy(x_mat).tolist()

    sim_data_dict_dir = '{}/sim_dat_test.json'.format(scenario_name_dir)
    with open(sim_data_dict_dir, "w") as write_file:
        json.dump(sim_data_dict, write_file)

    return sim_data_dict
