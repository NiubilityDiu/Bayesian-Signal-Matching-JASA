from self_py_fun.SimGlobal import *
import numpy as np
import math
import scipy.stats as stats
from scipy.linalg import solve
from scipy.stats import multivariate_normal as mvn
import jax.numpy as jnp
import json
from scipy import linalg
from sklearn.metrics import confusion_matrix, roc_auc_score


def import_sim_data(
        sim_data_dir, seq_i, seq_size_ls, N, E, signal_length_2, matrix_normal_bool=True
):
    r"""
    :param sim_data_dir:
    :param seq_i:
    :param seq_size_ls:
    :param N:
    :param E:
    :param signal_length_2:
    :param matrix_normal_bool: bool
    :return:
    """
    with open(sim_data_dir, 'r') as file:
        sim_data_dict = json.load(file)

    target_char = sim_data_dict['omega']
    target_char_len = len(target_char)

    for n in range(N):
        subject_i_name = 'subject_{}'.format(n)
        sim_data_dict[subject_i_name]['X'] = np.reshape(
            np.array(sim_data_dict[subject_i_name]['X']),
            [target_char_len * seq_size_ls[n] * rcp_unit_flash_num, E * signal_length_2]
        )
        sim_data_dict[subject_i_name]['Y'] = np.reshape(
            np.array(sim_data_dict[subject_i_name]['Y']),
            [target_char_len * seq_size_ls[n] * rcp_unit_flash_num]
        )

    input_data = {}
    for n in range(N):
        subject_n_name = 'subject_{}'.format(n)
        sim_data_subject_n = sim_data_dict[subject_n_name]
        input_data[subject_n_name] = {
            'target': sim_data_subject_n['X'][sim_data_subject_n['Y'] == 1, :],
            'non-target': sim_data_subject_n['X'][sim_data_subject_n['Y'] == 0, :]
        }

    # subset the data for new participant (0)
    subject_0_tar_rs = np.reshape(
        input_data['subject_0']['target'], [target_char_len, seq_size_ls[0] * 2, E * signal_length_2]
    )
    tar_event_i_size = target_char_len * (seq_i + 1) * 2
    input_data['subject_0']['target'] = np.reshape(
        subject_0_tar_rs[:, :(seq_i + 1) * 2, :], [tar_event_i_size, E * signal_length_2]
    )
    subject_0_ntar_rs = np.reshape(
        input_data['subject_0']['non-target'], [target_char_len, seq_size_ls[0] * 10, E * signal_length_2]
    )
    ntar_event_i_size = target_char_len * (seq_i + 1) * 10
    input_data['subject_0']['non-target'] = np.reshape(
        subject_0_ntar_rs[:, :(seq_i + 1) * 10, :], [ntar_event_i_size, E * signal_length_2]
    )

    if E > 1 and matrix_normal_bool:
        input_data['subject_0']['target'] = np.reshape(
            input_data['subject_0']['target'], [tar_event_i_size, E, signal_length_2]
        )
        input_data['subject_0']['non-target'] = np.reshape(
            input_data['subject_0']['non-target'], [ntar_event_i_size, E, signal_length_2]
        )
        for n in range(N - 1):
            subject_n_name = 'subject_{}'.format(n + 1)
            tar_event_n_size = target_char_len * seq_size_ls[n + 1] * 2
            input_data[subject_n_name]['target'] = np.reshape(
                input_data[subject_n_name]['target'], [tar_event_n_size, E, signal_length_2]
            )
            ntar_event_n_size = target_char_len * seq_size_ls[n + 1] * 10
            input_data[subject_n_name]['non-target'] = np.reshape(
                input_data[subject_n_name]['non-target'], [ntar_event_n_size, E, signal_length_2]
            )

    new_data = input_data['subject_0']
    source_data = input_data.copy()
    source_data.pop('subject_0')  # efficient way to remove keys from an existing dictionary

    return source_data, new_data


def determine_row_column_indices(target_letter):
    target_letter = target_letter.upper()
    assert target_letter in sim_rcp_array, print('The input doesn\'t belong to the letter table.')
    letter = sim_rcp_array.index(target_letter) + 1
    assert 1 <= letter <= 36
    row_index = int(np.ceil(letter / 6))
    column_index = (letter + 6 - 1) % 6 + 6 + 1
    return row_index, column_index


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def compute_q_matrix(target_char, rcp_array: list):
    target_index = rcp_array.index(target_char)
    # print('Target_index = {}'.format(target_index))
    target_row = int(np.ceil((target_index + 1) / 6))
    target_col = target_index % 6 + 7

    # target, non-target
    q_matrix = np.concatenate(
        [np.zeros([rcp_unit_flash_num, 1]), np.ones([rcp_unit_flash_num, 1])], axis=1
    )
    q_matrix[target_row-1, :] = np.array([1, 0])
    q_matrix[target_col-1, :] = np.array([1, 0])

    # delta, non-target
    q_matrix_reparam = np.copy(q_matrix)
    q_matrix_reparam[target_row - 1, :] = np.array([1, 1])
    q_matrix_reparam[target_col - 1, :] = np.array([1, 1])

    return q_matrix, q_matrix_reparam


def compute_p_matrix(w_vector):
    p_matrix = np.zeros([rcp_unit_flash_num, rcp_unit_flash_num])
    for i in range(rcp_unit_flash_num):
        # print(w_vector[i])
        p_matrix[i, w_vector[i]-1] = 1
    return p_matrix


def convert_log_prob_to_prob(log_prob_vec):

    prob_vec_len = len(log_prob_vec)
    log_prob_max = np.max(log_prob_vec)
    log_prob_vec_2 = log_prob_vec - log_prob_max
    prob_vec = np.exp(log_prob_vec_2)
    prob_vec = prob_vec / np.sum(prob_vec, axis=0)
    prob_vec[prob_vec > 0.01] = prob_vec[prob_vec > 0.01] - 0.01
    prob_vec = np.around(prob_vec, decimals=2)

    prob_vec_descend = np.sort(prob_vec)[::-1]
    arg_prob_vec_descend = np.argsort(prob_vec)[::-1]
    # use 0.95 as a threshold
    n_keep_max = np.where(np.cumsum(prob_vec_descend > 0.95))[0]
    if len(n_keep_max) > 0:
        n_keep_max = n_keep_max[0]
        p_majority = 1 - np.sum(prob_vec_descend[:(n_keep_max + 1)])
        p_rest_val = p_majority / (prob_vec_len - n_keep_max - 1)
        arg_prob_vec_keep = arg_prob_vec_descend[:(n_keep_max + 1)]
        for iter_id in range(prob_vec_len):
            if iter_id not in arg_prob_vec_keep:
                prob_vec[iter_id] = p_rest_val
    return prob_vec


def sample_normal_from_canonical_form(lambda_mat, eta_vector, sample_size=1):
    regular_mean = solve(lambda_mat, eta_vector)
    lambda_mat_chol_inv = np.linalg.inv(np.linalg.cholesky(lambda_mat))
    regular_cov_mat = np.matmul(np.transpose(lambda_mat_chol_inv), lambda_mat_chol_inv)
    normal_sample = mvn(mean=regular_mean, cov=regular_cov_mat).rvs(size=sample_size)
    return normal_sample


def compute_P_k_matrix_multi(psi_k, eigen_fun_mat):
    # n_component = eigen_fun_mat.shape[1]
    D_psi_k = np.diag(psi_k)
    # P_k = np.matmul(np.kron(D_psi_k, np.eye(n_component)), np.kron(np.eye(E), eigen_fun_mat))
    # (x) operation rule:
    # A (x) B = (I2 (x) B)(A (x) I1) = (A (x) I1)(I2 (x) B)
    P_k = np.kron(D_psi_k, eigen_fun_mat)
    return P_k


def compute_cov_k_matrix_multi(sigma_k, rho_k, eta_k, signal_length, E):
    cov_k_temp = create_exponential_decay_cov_mat(1.0, rho_k, signal_length)
    cov_k_spatial = create_cs_cov_mat(sigma_k ** 2, eta_k, E)
    cov_k_kron = np.kron(cov_k_temp, cov_k_spatial)
    return cov_k_kron


def compute_cov_k_matrix_inv_multi(sigma_k, rho_k, eta_k, signal_length, E):
    # fast way to compute the inverse w.r.t kronecker product.
    # (A (x) B)^-1 = A^-1 (x) B^(-1)
    cov_k_temp = create_exponential_decay_cov_mat(1.0, rho_k, signal_length)
    cov_k_spatial = create_cs_cov_mat(sigma_k ** 2, eta_k, E)

    cov_chol_inv_k_temp = np.linalg.inv(np.linalg.cholesky(cov_k_temp))
    cov_inv_k_temp = np.matmul(np.transpose(cov_chol_inv_k_temp), cov_chol_inv_k_temp)
    cov_chol_inv_k_spatial = np.linalg.inv(np.linalg.cholesky(cov_k_spatial))
    cov_inv_k_spatial = np.matmul(np.transpose(cov_chol_inv_k_spatial), cov_chol_inv_k_spatial)
    cov_inv_k = np.kron(cov_inv_k_temp, cov_inv_k_spatial)
    return cov_inv_k


def generate_label_N_K_general(prob_ls, K):
    label_N_dict = {}
    sub_num = len(prob_ls)
    for sub_id in range(sub_num):
        assert len(prob_ls[sub_id]) == K-1 and np.sum(prob_ls[sub_id]) <= 1, print('Incorrect probability input!')
        sub_id_name = 'subject_{}'.format(sub_id)
        label_N_dict[sub_id_name] = np.append(prob_ls[sub_id], 1-np.sum(prob_ls[sub_id]))
    return label_N_dict


def permute_order_label_switch(input_vec, arg_max_index):
    K = len(input_vec)
    mat_group_0_left = np.eye(K)
    order_mag = np.append(0, np.argsort(input_vec[1:])[::-1] + 1)
    if arg_max_index != 0:
        order_group_0_left = [k for k in range(K)]
        order_group_0_left[arg_max_index], order_group_0_left[0] = \
            order_group_0_left[0], order_group_0_left[arg_max_index]
        mat_group_0_left = create_permutation_transform_matrix(order_group_0_left)
        input_vec_temp = np.copy(input_vec)
        input_vec_temp[arg_max_index] = input_vec_temp[0]
        order_mag = np.append(0, np.argsort(input_vec_temp[1:])[::-1] + 1)
    mat_order_mag = create_permutation_transform_matrix(order_mag)
    # mat_order_mag = np.eye(K)
    double_mat = np.matmul(mat_order_mag, mat_group_0_left)
    output_K_order = np.matmul(double_mat, np.arange(K))
    return double_mat, output_K_order


def generate_label_N_K_general_2(sub_num, K):
    label_N_dict = {}
    for sub_id in range(sub_num):
        sub_id_name = 'subject_{}'.format(sub_id)
        # label_N_dict[sub_id_name] = np.squeeze(mtn.rvs(n=1, p=np.ones(K)/K, size=1), axis=0)
        label_N_dict[sub_id_name] = np.ones(K)/K
    return label_N_dict


def create_permutation_transform_matrix(order_vector):
    mat_size = len(order_vector)
    output_mat = np.zeros([mat_size, mat_size])
    for mat_row in range(mat_size):
        output_mat[mat_row, order_vector[mat_row]] = 1
    return output_mat


def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):

    r"""
    sigma_sq: scalar_like,
    first_column_except_1: array_like, 1d-array, except diagonal 1.
    return:
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    """
    first_column = np.r_[1, first_column_except_1]
    del first_column_except_1
    assert first_column[0] == 1, print('the first entry should be 1!')
    cov_mat = sigma_sq * linalg.toeplitz(first_column)
    return cov_mat


def create_exponential_decay_cov_mat(sigma_sq, rho, mat_size):
    # temporal correlation matrix
    assert 0 <= rho < 1, print('rho ranges from 0 to 1.')
    if rho > 0:
        log_first_column_except_1 = np.arange(1, mat_size) * np.log(rho)
        first_column_except_1 = np.exp(log_first_column_except_1)
    else:
        first_column_except_1 = np.zeros(mat_size-1)
    cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    return cov_mat


def create_cs_cov_mat(sigma_sq, eta_val: float, mat_size: int):
    # spatial dependency matrix (compound symmetry)
    assert 0 <= eta_val < 1, print('spatial dependency assumes ranges between 0 and 1.')
    cov_mat = np.ones([mat_size, mat_size]) * eta_val + np.diag(np.zeros(mat_size) + 1 - eta_val)
    if isinstance(sigma_sq, float):
        cov_mat = sigma_sq * cov_mat
    else:
        V_k = np.diag(np.sqrt(sigma_sq))
        cov_mat = np.matmul(np.matmul(V_k, cov_mat), np.transpose(V_k))
    return cov_mat


def create_exponential_decay_cov_mat_jnp(sigma_sq, rho, mat_size):
    log_first_row = jnp.arange(mat_size) * jnp.log(rho)
    cov_mat = jnp.eye(mat_size)
    for mat_i in np.arange(1, mat_size):
        cov_mat = cov_mat + jnp.diagflat(jnp.exp(log_first_row[mat_i]) + jnp.zeros(mat_size-mat_i), mat_i) + \
                  jnp.diagflat(jnp.exp(log_first_row[mat_i]) + jnp.zeros(mat_size-mat_i), -mat_i)
    return sigma_sq * cov_mat


def create_cs_cov_mat_jnp(sigma, lambda_val, mat_size):
    r"""
    :param sigma: heterogeneous standard deviation term
    :param lambda_val:
    :param mat_size:
    :return:
    """
    sigma = jnp.array(sigma)
    cov_mat = jnp.diag(jnp.ones([mat_size]) * (1-lambda_val), k=0) + jnp.ones([mat_size, mat_size]) * lambda_val
    return jnp.matmul(jnp.matmul(jnp.diag(sigma), cov_mat), jnp.diag(sigma))


# squared exponential kernel with diagonal noise term
def kernel_squared_exp(index_x, index_z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((index_x[:, None] - index_z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(index_x.shape[0])
    return k


def kernel_gamma_exp(index_x, index_z, var, length, gamma_val, noise, jitter=1.0e-6, include_noise=True):
    assert 0 < gamma_val < 2, print('incorrect gamma value input!')
    deltaXsq = (jnp.abs((index_x[:, None] - index_z)) / length) ** gamma_val
    k = var * jnp.exp(-deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(index_x.shape[0])
    return k


def create_kernel_function(length, gamma_val, K, signal_length_inside):
    r"""
    :param length:
    :param gamma_val:
    :param K:
    :param signal_length_inside:
    :return:
    Notice that the var and noise are 1.0 and 0.0 by default, respectively.
    """
    index_x = (np.arange(signal_length_inside) - int(signal_length_inside / 2)) / signal_length_inside
    eigen_val_dict = {}
    eigen_fun_mat_dict = {}
    for k in range(K):
        kernel_gram_mat_ntar_k = kernel_gamma_exp(
            index_x, index_x, 1.0, length[k][0], gamma_val[k][0], 0
        )
        eigen_val_ntar_k, eigen_fun_ntar_k = kernel_mercer_representation(kernel_gram_mat_ntar_k)

        kernel_gram_mat_tar_k = kernel_gamma_exp(
            index_x, index_x, 1.0, length[k][1], gamma_val[k][1], 0
        )
        eigen_val_tar_k, eigen_fun_tar_k = kernel_mercer_representation(kernel_gram_mat_tar_k)

        group_k_name = 'group_{}'.format(k)
        eigen_val_dict[group_k_name] = {'target': eigen_val_tar_k, 'non-target': eigen_val_ntar_k}
        eigen_fun_mat_dict[group_k_name] = {'target': eigen_fun_tar_k, 'non-target': eigen_fun_ntar_k}

    return eigen_val_dict, eigen_fun_mat_dict


def kernel_mercer_representation(gram_mat):
    # check pdf
    if is_pos_def(gram_mat):
        mat_size, _ = gram_mat.shape
        eigen_val, eigen_vector = np.linalg.eig(gram_mat)
        # output 95% of eigen_val
        eigen_val_cumsum = np.cumsum(eigen_val)
        eigen_val_cumsum_percent = eigen_val_cumsum / eigen_val_cumsum[-1]

        threshold_num = np.where(eigen_val_cumsum_percent >= 0.95)[0][0]
        # print(eigen_val_cumsum_percent[threshold_num])
        print(threshold_num+1)
        return eigen_val[:(threshold_num+1)], eigen_vector[:, :(threshold_num+1)]
    else:
        return 'The input matrix is not positive definite.'


def compute_bhattacharyya_distance_normal(mu1, mu2, cov1, cov2):
    """
    :param mu1: 2d-array, (window_len, 1)
    :param mu2: 2d-array, (window_len, 1)
    :param cov1: square matrix, (window_len, window_len)
    :param cov2: square matrix, (window_len, window_len)
    :return: A scalar, notice that the function only applies to multivariate normal distribution.
    """
    feature_len = cov1.shape[0]
    ridge_alpha = 0.1 * np.ones([feature_len])
    cov1 = cov1 + np.diag(ridge_alpha)
    cov2 = cov2 + np.diag(ridge_alpha)
    cov = (cov1 + cov2) / 2
    cov_chol_inv = np.linalg.inv(np.linalg.cholesky(cov))
    cov_inv = np.matmul(np.transpose(cov_chol_inv), cov_chol_inv)
    mu_diff = mu1 - mu2
    quad_comp = 1/8 * np.matmul(np.matmul(np.transpose(mu_diff), cov_inv), mu_diff)
    quad_comp = np.squeeze(quad_comp)  # convert from (1, 1) to scalar

    _, logdet_cov = np.linalg.slogdet(cov)
    cov1_chol = np.linalg.cholesky(cov1)
    cov2_chol = np.linalg.cholesky(cov2)
    det_cov1_chol = np.linalg.det(cov1_chol)
    det_cov2_chol = np.linalg.det(cov2_chol)
    cov_comp = 1/2 * (logdet_cov - np.log(det_cov1_chol) - np.log(det_cov2_chol))

    return quad_comp + cov_comp


def compute_bhattacharyya_distance_matrix_normal(
        mu_mat_1, mu_mat_2, cov_temp_1, cov_temp_2,
        cov_spatial_1, cov_spatial_2, channel_dim, window_length
):
    """
    :param window_length: positive integer
    :param channel_dim: positive integer
    :param mu_mat_1: 2d-array: (E, window_len)
    :param mu_mat_2: 2d-array: (E, window_len)
    :param cov_temp_1: square matrix: (window_len, window_len)
    :param cov_temp_2: square matrix: (window_len, window_len)
    :param cov_spatial_1: square matrix: (E, E)
    :param cov_spatial_2: square matrix: (E, E)
    :return: distance based on the vectorization format of two matrix normal distributions
    """
    mu_1 = np.reshape(mu_mat_1, [window_length * channel_dim, 1])
    mu_2 = np.reshape(mu_mat_2, [window_length * channel_dim, 1])

    cov_1 = np.kron(cov_spatial_1, cov_temp_1)
    cov_2 = np.kron(cov_spatial_2, cov_temp_2)
    b_dist_value = compute_bhattacharyya_distance_normal(mu_1, mu_2, cov_1, cov_2)
    return b_dist_value


# prediction functions
def swlda_predict_binary_likelihood(
        b_inmodel, mu_tar, mu_ntar, std_common, signal_test
):
    pred_score = np.matmul(signal_test, b_inmodel)
    score_size = pred_score.shape[0]
    pred_score_binary = np.zeros_like(pred_score) - 1
    for i in range(score_size):
        log_pdf_tar_i = stats.norm.logpdf(pred_score[i, 0], loc=mu_tar, scale=std_common)
        log_pdf_ntar_i = stats.norm.logpdf(pred_score[i, 0], loc=mu_ntar, scale=std_common)
        if log_pdf_tar_i > log_pdf_ntar_i:
            pred_score_binary[i, 0] = 1
    return pred_score, pred_score_binary


def _ml_predict_letter_likelihood_unit(
        char_prob, stimulus_score, stimulus_code, mu_tar, mu_ntar, std_common, unit_stimulus_set
):
    """
    Apply the bayesian naive dynamic stopping criterion
    :param char_prob:
    :param stimulus_score:
    :param stimulus_code:
    :param mu_tar:
    :param mu_ntar:
    :param std_common:
    :param unit_stimulus_set:
    :return:
    """
    char_prob_post = np.copy(np.log(char_prob))
    for s_id in range(rcp_unit_flash_num):
        for char_id in range(1, rcp_char_size + 1):
            if char_id in unit_stimulus_set[stimulus_code[s_id]-1]:
                char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                         stats.norm.logpdf(stimulus_score[s_id], loc=mu_tar, scale=std_common)
            else:
                char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                         stats.norm.logpdf(stimulus_score[s_id], loc=mu_ntar, scale=std_common)
    char_prob_post = char_prob_post - np.max(char_prob_post)
    char_prob_post = np.exp(char_prob_post)
    char_prob_post = char_prob_post / np.sum(char_prob_post)
    return char_prob_post


def ml_predict_letter_likelihood(
        stimulus_score, stimulus_code, letter_dim, repet_pred,
        mu_tar, mu_ntar, std_common, unit_stimulus_set, letter_table_ls
):
    stimulus_score = np.reshape(stimulus_score, [letter_dim, repet_pred, rcp_unit_flash_num])
    stimulus_code = np.reshape(stimulus_code, [letter_dim, repet_pred, rcp_unit_flash_num])
    char_prob_mat = np.zeros([letter_dim, repet_pred+1, rcp_char_size]) + 1 / rcp_char_size
    for letter_id in range(letter_dim):
        for seq_id in range(repet_pred):
            char_prob_mat[letter_id, seq_id+1, :] = _ml_predict_letter_likelihood_unit(
                char_prob_mat[letter_id, seq_id, :], stimulus_score[letter_id, seq_id, :],
                stimulus_code[letter_id, seq_id, :], mu_tar, mu_ntar, std_common, unit_stimulus_set
            )
    char_prob_mat = char_prob_mat[:, 1:, :]
    argmax_prob_mat = np.argmax(char_prob_mat, axis=-1)
    char_max_mat = np.zeros_like(argmax_prob_mat).astype('<U5')
    for letter_id in range(letter_dim):
        for seq_id in range(repet_pred):
            char_max_mat[letter_id, seq_id] = letter_table_ls[argmax_prob_mat[letter_id, seq_id]]
    return char_max_mat, char_prob_mat


def compute_prediction_accuracy_inside(
        true_char_arr, predict_char_arr_cum
):
    """
    :param true_char_arr:
    :param predict_char_arr_cum:
    :return:
    We only look at prediction results from the last sequence.
    """
    predict_cum_dict = {}
    total_char_cum = len(true_char_arr)
    predict_cum_dict['Total'] = int(total_char_cum)
    total_accuracy = 0
    for char_id in range(total_char_cum):
        # Remove extra space in the string for comparison
        predict_char_id = predict_char_arr_cum[char_id, -1].strip()
        true_char_id = true_char_arr[char_id].strip()
        accuracy_id = int(predict_char_id in true_char_id)
        # if 'BS' in predict_char_id and 'BS' in true_char_id:
        #     accuracy_id = 1
        # elif predict_char_id == 'SPACE' and true_char_id == '':
        #     accuracy_id = 1
        # print(accuracy_id)
        total_accuracy = total_accuracy + accuracy_id
    predict_cum_dict['Accuracy'] = total_accuracy

    return predict_cum_dict


def summarize_accuracy_prob(
        b_inmodel, mu_tar, mu_ntar, std_common,
        signal_test, code_test, true_letter, seq_test,
        unit_stimulus_set, letter_table_ls
):
    pred_prob_arr, pred_binary_arr = swlda_predict_binary_likelihood(
        b_inmodel, mu_tar, mu_ntar, std_common, signal_test
    )

    # for first sequence, swLDA may be unreliable.
    if math.isnan(pred_prob_arr[0, 0]) or std_common == 0:
        pred_prob_letter = 'A'
        pred_prob_mat = np.ones([1, 1, 36]) / 36
    else:
        pred_prob_letter, pred_prob_mat = ml_predict_letter_likelihood(
            pred_prob_arr, code_test, len(true_letter), seq_test, mu_tar, mu_ntar, std_common,
            unit_stimulus_set, letter_table_ls
        )
    return pred_prob_letter, pred_prob_mat


def summarize_accuracy_likelihood(
        b_inmodel, mu_tar, mu_ntar, std_common,
        signal_test, label_test, code_test,
        sub_name, true_letter, seq_test, seq_train,
        unit_stimulus_set, letter_table_ls, decision_rule
):
    pred_prob_arr, pred_binary_arr = swlda_predict_binary_likelihood(
        b_inmodel, mu_tar, mu_ntar, std_common, signal_test
    )
    # binary classification
    pred_binary_accuracy = np.mean(pred_binary_arr == label_test)
    tn, fp, fn, tp = confusion_matrix(label_test, pred_binary_arr).ravel()
    auc_val = roc_auc_score(label_test, pred_prob_arr)

    pred_prob_letter, pred_prob_mat = ml_predict_letter_likelihood(
        pred_prob_arr, code_test, len(true_letter), seq_test, mu_tar, mu_ntar, std_common,
        unit_stimulus_set, letter_table_ls
    )
    letter_accuracy = compute_prediction_accuracy_inside(
        true_letter, pred_prob_letter
    )
    print(letter_accuracy)

    result_dict = {
        'ID': sub_name,
        'num_letter': len(true_letter),
        'seq_test': seq_test,
        'seq_train': seq_train,
        'binary': pred_binary_accuracy,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'auc': auc_val,
        'letter': letter_accuracy['Accuracy'] / letter_accuracy['Total'],
        'rule': decision_rule
    }

    return pred_prob_arr, pred_binary_arr, result_dict



