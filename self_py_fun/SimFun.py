# Functions for simulation studies.
import os
from self_py_fun.SimGlobal import *
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial as mtn
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import json
import seaborn as sns
import torch
from scipy import linalg
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import TransformReparam
import scipy.io as sio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


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


def generate_label_N_K_general(prob_ls, K):
    label_N_dict = {}
    sub_num = len(prob_ls)
    for sub_id in range(sub_num):
        assert len(prob_ls[sub_id]) == K-1 and np.sum(prob_ls[sub_id]) <= 1, print('Incorrect probability input!')
        sub_id_name = 'subject_{}'.format(sub_id)
        label_N_dict[sub_id_name] = np.append(prob_ls[sub_id], 1-np.sum(prob_ls[sub_id]))
    return label_N_dict


def generate_label_N_K_general_2(sub_num, K):
    label_N_dict = {}
    for sub_id in range(sub_num):
        sub_id_name = 'subject_{}'.format(sub_id)
        # label_N_dict[sub_id_name] = np.squeeze(mtn.rvs(n=1, p=np.ones(K)/K, size=1), axis=0)
        label_N_dict[sub_id_name] = np.ones(K)/K
    return label_N_dict


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


def create_exponential_decay_cov_mat(sigma_sq: float, rho: float, mat_size: int):
    assert 0 <= rho <= 1, print('rho ranges from 0 to 1.')
    if rho > 0:
        log_first_column_except_1 = np.arange(1, mat_size) * np.log(rho)
        first_column_except_1 = np.exp(log_first_column_except_1)
    else:
        first_column_except_1 = np.zeros(mat_size-1)
    cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    return cov_mat


def create_cs_cov_mat(sigma_sq, lambda_val: float, mat_size: int):
    assert 0 <= lambda_val <= 1, print('spatial dependency assumes ranges between 0 and 1.')
    cov_mat = np.ones([mat_size, mat_size]) * lambda_val + np.diag(np.zeros(mat_size)+1-lambda_val)
    if isinstance(sigma_sq, float):
        cov_mat = sigma_sq * cov_mat
    else:
        cov_mat = np.matmul(np.matmul(np.diag(np.sqrt(sigma_sq)), cov_mat), np.diag(np.sqrt(sigma_sq)))
    return cov_mat


def create_exponential_decay_cov_mat_jnp(sigma_sq, rho, mat_size):
    log_first_row = jnp.arange(mat_size) * jnp.log(rho)
    cov_mat = jnp.eye(mat_size)
    for mat_i in np.arange(1, mat_size):
        cov_mat = cov_mat + jnp.diagflat(jnp.exp(log_first_row[mat_i]) + jnp.zeros(mat_size-mat_i), mat_i) + \
                  jnp.diagflat(jnp.exp(log_first_row[mat_i]) + jnp.zeros(mat_size-mat_i), -mat_i)
    return sigma_sq * cov_mat


def create_cs_cov_mat_jnp(sigma_sq, lambda_val, mat_size):
    r"""
    :param sigma_sq: heterogeneous variance term
    :param lambda_val:
    :param mat_size:
    :return:
    """
    sigma_vec = jnp.sqrt(jnp.array(sigma_sq))
    cov_mat = jnp.diag(jnp.ones([mat_size]) * (1-lambda_val), k=0) + jnp.ones([mat_size, mat_size]) * lambda_val
    return jnp.matmul(jnp.matmul(jnp.diag(sigma_vec), cov_mat), jnp.diag(sigma_vec))


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


def kernel_mercer_representation(gram_mat):
    # check pdf
    if is_pos_def(gram_mat):
        mat_size, _ = gram_mat.shape
        eigen_val, eigen_vector = np.linalg.eig(gram_mat)
        # output 95% of eigen_val
        eigen_val_cumsum = np.cumsum(eigen_val)
        eigen_val_cumsum_percent = eigen_val_cumsum / eigen_val_cumsum[-1]

        threshold_num = np.where(eigen_val_cumsum_percent >= 0.95)[0][0]
        print(eigen_val_cumsum_percent[threshold_num])
        # print(threshold_num+1)
        return eigen_val[:(threshold_num+1)], eigen_vector[:, :(threshold_num+1)]
    else:
        return 'The input matrix is not positive definite.'


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
    # self.save_eeg_ml_predict_likelihood(
    #     pred_prob_letter, local_bool, sub_name, method_name,
    #     'Prediction_{}'.format(frt_file_name)
    # )
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

    # if local_bool:
    #     parent_path = '{}/FRT_files'.format(parent_path_local)
    # else:
    #     parent_path = '{}/FRT_files'.format(parent_path_slurm)
    # if not os.path.exists('{}/{}/{}'.format(parent_path, sub_name, method_name)):
    #     os.mkdir('{}/{}/{}'.format(parent_path, sub_name, method_name))
    return pred_prob_arr, pred_binary_arr, result_dict

