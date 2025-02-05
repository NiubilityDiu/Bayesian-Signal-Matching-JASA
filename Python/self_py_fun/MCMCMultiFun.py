from self_py_fun.MCMCMultiRefFun import *
from self_py_fun.ExistMLFun import *
from scipy.stats import multinomial as mtn
from scipy.stats import invgamma, expon, halfcauchy, lognorm
import scipy.io as sio


# multi-channel functions
def predict_reference_fast_multi(
        mcmc_output_dict, E, signal_length_2,
        new_subject_signal, eigen_fun_mat
):
    # only consider posterior mean to make a fast approximation
    beta_tar_0 = []
    beta_ntar_0 = []
    sigma_0 = []

    psi_tar_0 = np.squeeze(mcmc_output_dict['psi_tar_0'])
    psi_ntar_0 = np.squeeze(mcmc_output_dict['psi_ntar_0'])

    for e in range(E):
        alpha_tar_0_e = mcmc_output_dict['alpha_tar_0_{}'.format(e)]
        alpha_ntar_0_e = mcmc_output_dict['alpha_ntar_0_{}'.format(e)]

        sigma_0_e = mcmc_output_dict['sigma_0_{}'.format(e)]

        beta_tar_0_e = psi_tar_0[:, np.newaxis] * alpha_tar_0_e @ eigen_fun_mat['target'].T
        beta_ntar_0_e = psi_ntar_0[:, np.newaxis] * alpha_ntar_0_e @ eigen_fun_mat['non-target'].T

        beta_tar_0.append(beta_tar_0_e)
        beta_ntar_0.append(beta_ntar_0_e)
        sigma_0.append(sigma_0_e)

    beta_tar_0 = np.stack(beta_tar_0, axis=1)
    beta_ntar_0 = np.stack(beta_ntar_0, axis=1)
    sigma_0 = np.transpose(np.concatenate(sigma_0, axis=0))  # (200, E)

    rho_0 = mcmc_output_dict['rho_0']
    eta_0 = mcmc_output_dict['eta_0']

    beta_tar_0_mean = np.mean(beta_tar_0, axis=0)
    beta_ntar_0_mean = np.mean(beta_ntar_0, axis=0)
    sigma_0_mean = np.mean(sigma_0, axis=0)
    rho_0_mean = np.mean(rho_0)
    eta_0_mean = np.mean(eta_0)

    matn_tar_obj = create_mat_normal_obj_from_mcmc(
        beta_tar_0_mean, rho_0_mean, eta_0_mean, sigma_0_mean, signal_length_2, E
    )
    matn_ntar_obj = create_mat_normal_obj_from_mcmc(
        beta_ntar_0_mean, rho_0_mean, eta_0_mean, sigma_0_mean, signal_length_2, E
    )

    log_lhd_tar_0 = matn_tar_obj.logpdf(X=new_subject_signal)
    log_lhd_ntar_0 = matn_ntar_obj.logpdf(X=new_subject_signal)

    # score_0 = 1 / (1 + np.exp(log_lhd_ntar_0 - log_lhd_tar_0))
    return log_lhd_tar_0 - log_lhd_ntar_0, log_lhd_tar_0, log_lhd_ntar_0


def predict_cluster_fast_multi(
        mcmc_output_dict, E, signal_length_2,
        new_subject_signal, mcmc_ids
):
    # only consider posterior mean to make a fast approximation
    # remove outliers based on sigma values for group 1 to group K-1.
    if 'B_tar' in mcmc_output_dict.keys():
        if mcmc_ids is not None:
            B_tar = mcmc_output_dict['B_tar'][mcmc_ids, ...]
            B_0_ntar = mcmc_output_dict['B_0_ntar'][mcmc_ids, ...]
            rho = mcmc_output_dict['rho'][mcmc_ids, :]
            sigma = mcmc_output_dict['sigma'][mcmc_ids, ...]
            eta = mcmc_output_dict['eta'][mcmc_ids, :]
            z_vec = mcmc_output_dict['z_vector'][mcmc_ids, :]
        else:
            B_tar = mcmc_output_dict['B_tar']
            B_0_ntar = mcmc_output_dict['B_0_ntar']
            rho = mcmc_output_dict['rho']
            sigma = mcmc_output_dict['sigma']
            eta = mcmc_output_dict['eta']
            z_vec = mcmc_output_dict['z_vector']
    else:
        # multiple chains
        B_tar = []
        B_0_ntar = []
        rho = []
        sigma = []
        eta = []
        z_vec = []
        chain_name_ls = list(mcmc_output_dict.keys())[3:]
        # remove the first three names by default
        for chain_name_iter in chain_name_ls:
            B_tar.append(mcmc_output_dict[chain_name_iter]['B_tar'])
            B_0_ntar.append(mcmc_output_dict[chain_name_iter]['B_0_ntar'])
            rho.append(mcmc_output_dict[chain_name_iter]['rho'])
            sigma.append(mcmc_output_dict[chain_name_iter]['sigma'])
            eta.append(mcmc_output_dict[chain_name_iter]['eta'])
            z_vec.append(mcmc_output_dict[chain_name_iter]['z_vector'])

        B_tar = np.concatenate(B_tar, axis=0)
        B_0_ntar = np.concatenate(B_0_ntar, axis=0)
        rho = np.concatenate(rho, axis=0)
        sigma = np.concatenate(sigma, axis=0)
        eta = np.concatenate(eta, axis=0)
        z_vec = np.concatenate(z_vec, axis=0)

    # print('I am here!')
    B_0_tar = B_tar[np.sum(z_vec, axis=1)>0, 0, ...]
    B_0_ntar = B_0_ntar[np.sum(z_vec, axis=1)>0, ...]
    sigma_0 = sigma[np.sum(z_vec, axis=1)>0, 0, :]
    rho_0 = rho[np.sum(z_vec, axis=1)>0, 0]
    eta_0 = eta[np.sum(z_vec, axis=1)>0, 0]

    '''
    B_0_tar = B_tar[:, 0, ...]
    # B_0_ntar = B_0_ntar[np.sum(z_vec, axis=1)>0, ...]
    sigma_0 = sigma[:, 0, :]
    rho_0 = rho[:, 0, :]
    eta_0 = eta[:, 0, :]
    '''

    B_0_tar_mean = np.mean(B_0_tar, axis=0)
    B_0_ntar_mean = np.mean(B_0_ntar, axis=0)
    sigma_0_mean = np.mean(sigma_0, axis=0)
    rho_0_mean = np.mean(rho_0, axis=0)
    eta_0_mean = np.mean(eta_0, axis=0)

    matn_tar_obj = create_mat_normal_obj_from_mcmc(
        B_0_tar_mean, rho_0_mean, eta_0_mean, sigma_0_mean, signal_length_2, E
    )
    matn_ntar_obj = create_mat_normal_obj_from_mcmc(
        B_0_ntar_mean, rho_0_mean, eta_0_mean, sigma_0_mean, signal_length_2, E
    )

    log_lhd_ntar_0 = matn_tar_obj.logpdf(X=new_subject_signal)
    log_lhd_tar_0 = matn_ntar_obj.logpdf(X=new_subject_signal)
    # score_0 = 1 / (1 + np.exp(log_lhd_ntar_0 - log_lhd_tar_0))
    return log_lhd_tar_0 - log_lhd_ntar_0, log_lhd_tar_0, log_lhd_ntar_0


def predict_char_accuracy_multi(
        signal_input, type_input, code_input, mcmc_dict,
        seq_i_val, E_val, signal_len_val, mcmc_id_val,
        target_char_size_len, method_name_char, **kwargs
):
    exist_ml_obj = None
    if method_name_char == 'ref':
        pred_score_0_i, _, _ = predict_reference_fast_multi(
            mcmc_dict, E_val, signal_len_val, signal_input, mcmc_id_val
        )
        # eigen_fun_mat is the mcmc_id_val here for consistency (although not a good coding practice)
    elif method_name_char == 'cluster':
        pred_score_0_i, _, _ = predict_cluster_fast_multi(
            mcmc_dict, E_val, signal_len_val, signal_input, mcmc_id_val
        )
    elif method_name_char == 'xdawn_lda':
        n_components = kwargs['n_components']
        exist_ml_obj = kwargs['xdawn_lda_obj']
        if exist_ml_obj is None:
            exist_ml_obj = predict_xdawn_lda_fast_multi(
                n_components, E_val, signal_input, type_input
            )
        pred_score_0_i = exist_ml_obj.predict_log_proba(signal_input)[:, 1]
    else:
        domain_tradeoff_val = kwargs['domain_tradeoff']
        tol_val = kwargs['tol']
        exist_ml_obj = {}
        mdwm_obj = kwargs['mdwm_obj']
        if mdwm_obj is None:
            new_data_input = kwargs['new_data']
            old_data_input = kwargs['source_data']
            old_sub_name_ls = kwargs['source_sub_name']
            [mdwm_obj, mdwm_ERP_cov_obj, X_domain_mdwm_size, _] = (
                mdwm_fit_from_eeg_feature(
                    new_data_input, old_data_input, old_sub_name_ls,
                    domain_tradeoff_val, tol_val
            ))
        else:
            mdwm_ERP_cov_obj = kwargs['mdwm_ERP_cov_obj']
            X_domain_mdwm_size = kwargs['X_domain_mdwm_size']
        signal_input_cov = (mdwm_ERP_cov_obj.transform(signal_input) +
                            np.eye(X_domain_mdwm_size) * tol_val)
        pred_score_0_i = np.log(mdwm_obj.predict_proba(signal_input_cov)[:, 0])
        exist_ml_obj['mdwm_obj'] = mdwm_obj
        exist_ml_obj['mdwm_ERP_cov_obj'] = mdwm_ERP_cov_obj
    if type_input is None:
        # testing data only,
        # we do not need type_input to compute mu_tar, mu_ntar, and std_common
        mu_tar = kwargs['mu_tar']
        mu_ntar = kwargs['mu_ntar']
        std_common = kwargs['std_common']
    else:
        # for training data
        mu_tar = np.mean(pred_score_0_i[type_input == 1])
        mu_ntar = np.mean(pred_score_0_i[type_input != 1])
        std_common = np.std(pred_score_0_i)
        print('{}, {}, {}'.format(mu_tar, mu_ntar, std_common))

    # seq_i_val should be the correct one (not confused with index 0)
    pred_prob_letter, pred_prob_mat = ml_predict_letter_likelihood(
        pred_score_0_i, code_input, target_char_size_len,
        seq_i_val, mu_tar, mu_ntar, std_common,
        stimulus_group_set, sim_rcp_array
    )
    pred_prob_dict = {
        'letter': pred_prob_letter.tolist(),
        'prob': pred_prob_mat.tolist(),
    }
    return pred_prob_dict, exist_ml_obj, mu_tar, mu_ntar, std_common


def signal_integration_beta_multi_summary(
        mcmc_dict, K, eigen_fun_mat_dict, channel_dim
):
    beta_ntar_summary_dict = {}
    beta_tar_summary_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        beta_tar_k_dict, beta_ntar_k_dict = signal_beta_multi_ref_summary(
            mcmc_dict, eigen_fun_mat_dict[group_k_name], str(k), channel_dim, 0.05
        )
        beta_tar_summary_dict[group_k_name] = beta_tar_k_dict
        beta_ntar_summary_dict[group_k_name] = beta_ntar_k_dict

    return beta_tar_summary_dict, beta_ntar_summary_dict


def predict_hybrid_single_seq_multi(
    char_prob_seq_mat_pre, mcmc_c_dict, mcmc_r_dict, param_c, param_r,
    E, signal_length_2, eigen_fun_mat, unit_stimulus_set,
    new_seq_signal, new_seq_code
):
    pred_c_score_0, log_lhd_tar_c, log_lhd_ntar_c = predict_cluster_fast_multi(
        mcmc_c_dict, E, signal_length_2, new_seq_signal, None
    )
    log_lhd_sum_c = compute_single_seq_log_lhd_sum(pred_c_score_0, log_lhd_tar_c, log_lhd_ntar_c)

    pred_r_score_0, log_lhd_tar_r, log_lhd_ntar_r = predict_reference_fast_multi(
        mcmc_r_dict, E, signal_length_2, eigen_fun_mat, new_seq_signal
    )
    log_lhd_sum_r = compute_single_seq_log_lhd_sum(pred_r_score_0, log_lhd_tar_r, log_lhd_ntar_r)

    pred_score_final = np.copy(pred_r_score_0)
    mu_tar_final, mu_ntar_final, std_common_final = param_r
    if log_lhd_sum_c > log_lhd_sum_r:
        print('Clustering scores are used.')
        pred_score_final = np.copy(pred_c_score_0)
        mu_tar_final, mu_ntar_final, std_common_final = param_c
    else:
        print('Reference scores are used.')
    char_log_prob_seq_mat = ml_predict_letter_log_prob_single_seq(char_prob_seq_mat_pre, pred_score_final, new_seq_code,
                                                                  mu_tar_final, mu_ntar_final, std_common_final,
                                                                  unit_stimulus_set)
    return char_log_prob_seq_mat


def select_proper_mcmc_output_dict(
        source_name_ls, scenario_dir, xdawn_bool,
        n, seq_source_size_2, letter_dim_sub, **kwargs
):
    if source_name_ls is None:
        # simulation study scenario
        mcmc_output_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
            scenario_dir, n, seq_source_size_2
        )
    else:
        # EEG real data scenario
        if xdawn_bool:
            mcmc_output_dict_dir = select_xdawn_ref_target_mcmc_dict_dir(
                source_name_ls, letter_dim_sub, seq_source_size_2,
                n, scenario_dir, **kwargs
            )
        else:
            select_channel_ids_str = kwargs['select_channel_ids_str']
            # letter_dim_sub = kwargs['letter_dim_sub']
            mcmc_output_dict_dir = '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                    scenario_dir, source_name_ls[n], letter_dim_sub, select_channel_ids_str,
                    seq_source_size_2)
    mcmc_output_dict_result = sio.loadmat(mcmc_output_dict_dir)

    return mcmc_output_dict_result


def update_multi_gibbs_sampler_per_iteration(
        eigen_fun_mat_dict, eigen_val_dict,
        source_data, new_data, N_total, E, signal_length_2, source_name_ls,
        step_size_ls, estimate_eta_bool, estimate_rho_bool,
        scenario_dir, seq_source_size_2, seq_i,
        xdawn_bool, letter_dim_sub,
        *args, **kwargs
):

    B_tar_iter, B_0_ntar_iter, A_tar_iter, A_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter, \
    sigma_iter, rho_iter, eta_iter, z_vector_iter = args

    # update cluster-specific parameters
    psi_tar_accept = []
    psi_0_ntar_accept = []
    sigma_accept = []
    accept_iter_ls = []

    B_tar_iter = np.zeros_like(B_tar_iter)
    B_0_ntar_iter = np.zeros_like(B_0_ntar_iter)

    eigen_fun_mat_tar = eigen_fun_mat_dict['target']
    eigen_fun_mat_ntar = eigen_fun_mat_dict['non-target']
    eigen_val_tar = eigen_val_dict['target']
    eigen_val_ntar = eigen_val_dict['non-target']

    for n in range(N_total):
        # update alpha_k_tar and alpha_k_ntar
        # Here, we assume that alpha_k_tar_iter has the same dimension across k,
        # it may be different. In that case, we need to save alpha_k_tar_iter separately. (*)
        A_tar_iter = update_A_n_tar_multi(A_tar_iter, psi_tar_iter, sigma_iter, rho_iter, eta_iter,
                                          z_vector_iter, eigen_fun_mat_tar, eigen_val_tar, source_data,
                                          new_data, N_total, E, n, signal_length_2,
                                          scenario_dir, seq_source_size_2,
                                          xdawn_bool, letter_dim_sub, source_name_ls,
                                          **kwargs)

        # update psi_tar and psi_0_ntar
        psi_tar_iter, psi_n_tar_accept = update_psi_n_tar_multi_RWMH(
            A_tar_iter, psi_tar_iter, sigma_iter, rho_iter,
            eta_iter, z_vector_iter, eigen_fun_mat_tar,
            source_data, new_data, N_total, E, n, signal_length_2, step_size_ls[0][n],
            scenario_dir, seq_source_size_2,
            xdawn_bool, letter_dim_sub,
            source_name_ls, **kwargs
        )
        B_tar_iter[n, ...] = psi_tar_iter[n] * A_tar_iter[n, ...] @ eigen_fun_mat_tar.T

        if n == 0:
            A_0_ntar_iter = update_A_0_ntar_multi(psi_0_ntar_iter, sigma_iter, rho_iter, eta_iter,
                                                  eigen_fun_mat_ntar, eigen_val_ntar, new_data, E, signal_length_2)
            psi_0_ntar_iter, psi_0_ntar_accept = update_psi_0_ntar_multi_RWMH(
                A_0_ntar_iter, psi_0_ntar_iter, sigma_iter, rho_iter,
                eta_iter, eigen_fun_mat_ntar, new_data, E, signal_length_2, step_size_ls[2],
                **kwargs
            )
            B_0_ntar_iter = psi_0_ntar_iter * A_0_ntar_iter @ eigen_fun_mat_ntar.T

        # update sigma_k_iter
        sigma_iter, sigma_n_accept = update_sigma_n_multi_RWMH(B_tar_iter, B_0_ntar_iter, sigma_iter,
                                                               rho_iter, eta_iter, z_vector_iter, source_data,
                                                               new_data, N_total, E, n, signal_length_2,
                                                               scenario_dir, seq_source_size_2,
                                                               step_size_ls[1][n, :], xdawn_bool, letter_dim_sub,
                                                               source_name_ls, eigen_fun_mat_dict, **kwargs)

        if estimate_eta_bool:
            eta_iter = update_eta_n_multi_discrete_uniform(
                B_tar_iter, B_0_ntar_iter, sigma_iter, rho_iter, eta_iter, z_vector_iter,
                source_data, new_data, N_total, E, n, signal_length_2,
                scenario_dir, seq_source_size_2, xdawn_bool, letter_dim_sub,
                source_name_ls, **kwargs
            )
        else:
            # for real data analysis, since xDAWN makes each component orthogonal, and
            # all eta estimates are below 0.10.
            # eta can be set to 0.
            eta_iter = np.zeros(N_total)

        if estimate_rho_bool:
            rho_iter = update_rho_n_multi_discrete_uniform(B_tar_iter, B_0_ntar_iter,
                                                           sigma_iter, rho_iter, eta_iter, z_vector_iter,
                                                           source_data, new_data, N_total, E, n, signal_length_2,
                                                           scenario_dir, seq_source_size_2,
                                                           xdawn_bool, letter_dim_sub,
                                                           source_name_ls, **kwargs)
        else:
            # for real data analysis, since the original data pre-processing contains overlapping,
            # all rho estimates are over 0.8.
            # rho can be fixed as 0.85.
            rho_iter = np.zeros(N_total) + 0.85

        psi_tar_accept.append(psi_n_tar_accept)
        sigma_accept.append(sigma_n_accept)

    psi_tar_accept = np.stack(psi_tar_accept, axis=0)
    sigma_accept = np.stack(sigma_accept, axis=0)

    # update indicator Z
    z_vector_iter = update_Z_indicator_multi(B_tar_iter, sigma_iter, rho_iter, eta_iter, N_total, E, signal_length_2,
                                             source_name_ls, source_data, **kwargs)

    accept_iter_ls.append(psi_tar_accept)
    accept_iter_ls.append(sigma_accept)
    accept_iter_ls.append(psi_0_ntar_accept)

    log_joint_prob = compute_joint_log_likelihood_cluster_multi(
        B_tar_iter, B_0_ntar_iter, A_tar_iter, A_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter, sigma_iter,
        rho_iter, eta_iter, z_vector_iter,
        eigen_val_dict, source_data, new_data, N_total,
        E, signal_length_2, source_name_ls, **kwargs
    )

    return [[B_tar_iter, B_0_ntar_iter, A_tar_iter, A_0_ntar_iter,
             psi_tar_iter, psi_0_ntar_iter,
             sigma_iter, rho_iter, eta_iter, z_vector_iter],
            accept_iter_ls, log_joint_prob]


def update_A_n_tar_multi(
        A_tar, psi_tar, sigma, rho, eta, z_vector,
        eigen_fun_mat_tar, eigen_val_tar,
        source_data, new_data, N_total, E, n, signal_length_2,
        scenario_dir, seq_source_size_2,
        xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    r"""
    :param A_tar: 3d-array, (N+1, E, eigen_val_length)
    :param psi_tar: scalar value
    :param sigma: 2d-array, (N+1, E)
    :param rho: 1d-array with size (N+1)
    :param eta: 1d-array with size (N+1)
    :param z_vector: N-dim vector
    :param eigen_fun_mat_tar:
    :param eigen_val_tar:
    :param source_data:
    :param new_data:
    :param N_total:
    :param E:
    :param n:
    :param signal_length_2:
    :param scenario_dir:
    :param seq_source_size_2: integer
    :param source_name_ls:
    :param xdawn_bool: bool
    :param letter_dim_sub: integer
    :return:
    """

    eigen_val_tar_len = len(eigen_val_tar)

    if n == 0:
        # initialization
        Lambda_n_tar_temp = np.diag(eigen_val_tar)
        U_n_tar_temp = np.zeros([eigen_val_tar_len, E])

        psi_n_tar = psi_tar[n]
        sigma_n = sigma[n, :]
        eta_n = eta[n]
        rho_n = rho[n]

        size_n_tar = 0
        x_mat_n_tar_sum = np.zeros([signal_length_2, E])
        new_data_tar = new_data['target']
        size_n_tar = size_n_tar + new_data_tar.shape[0]
        x_mat_n_tar_sum = x_mat_n_tar_sum + np.sum(new_data_tar, axis=0).T

        corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
        corr_n_t_inv = compute_inverse_matrix(corr_n_t)

        for nn in range(N_total - 1):
            if z_vector[nn] == 1:
                if source_name_ls is None:
                    subject_nn_name = 'subject_{}'.format(nn + 1)
                else:
                    subject_nn_name = source_name_ls[nn + 1]
                source_data_nn_tar = source_data[subject_nn_name]['target']
                size_n_tar = size_n_tar + source_data_nn_tar.shape[0]
                x_mat_n_tar_sum = x_mat_n_tar_sum + np.sum(source_data_nn_tar, axis=0).T

        Lambda_n_tar_temp = Lambda_n_tar_temp + psi_n_tar ** 2 * \
                            size_n_tar * eigen_fun_mat_tar.T @ corr_n_t_inv @ eigen_fun_mat_tar
        U_n_tar_temp = U_n_tar_temp + psi_n_tar * eigen_fun_mat_tar.T @ corr_n_t_inv @ x_mat_n_tar_sum
        Lambda_n_tar = compute_inverse_matrix(Lambda_n_tar_temp)
        U_n_tar = (Lambda_n_tar @ U_n_tar_temp).T
        cov_n_s = create_cs_cov_mat(sigma_n ** 2, eta_n, E)
        A_n_tar_sample = generate_mat_normal_rvs_fast(U_n_tar, cov_n_s, Lambda_n_tar)

    else:
        ref_mcmmc_bool = kwargs['ref_mcmc_bool']
        ref_mcmc_id = kwargs['ref_mcmc_id']

        if ref_mcmmc_bool:
            # directly use numpyro results
            A_n_tar_sample = []
            mcmc_output_dict = select_proper_mcmc_output_dict(
                source_name_ls, scenario_dir, xdawn_bool, n,
                seq_source_size_2, letter_dim_sub, **kwargs
            )
            if ref_mcmc_id is None:
                for e in range(E):
                    A_n_tar_sample.append(np.mean(mcmc_output_dict['alpha_tar_0_{}'.format(e)], axis=0))
            else:
                for e in range(E):
                    A_n_tar_sample.append(mcmc_output_dict['alpha_tar_0_{}'.format(e)][ref_mcmc_id, :])
            A_n_tar_sample = np.stack(A_n_tar_sample, axis=0)
        else:
            # n ranges from 0 to 23, however, z_vector is a 23-dim vector.
            # here n is positive
            if z_vector[n-1] == 1:
                A_n_tar_sample = np.copy(A_tar[0, :])  # assume A_n_tar = A_0_tar, n>0 if z_n=1
            else:
                if source_name_ls is None:
                    # source subject id starts from 1 for simulation studies
                    subject_n_name = 'subject_{}'.format(n)
                else:
                    # source subject name starts from the second one.
                    subject_n_name = source_name_ls[n]
                A_n_tar_sample = update_A_n_tar_ref_multi(
                    psi_tar[n], sigma[n, :], rho[n], eta[n],
                    eigen_fun_mat_tar, eigen_val_tar, source_data[subject_n_name], E, signal_length_2
                )

    A_tar[n, ...] = np.copy(A_n_tar_sample)

    return A_tar


def update_A_n_tar_ref_multi(
        psi_n_tar, sigma_n, rho_n, eta_n,
        eigen_fun_mat_tar, eigen_val_tar,
        source_data_n, E, signal_length_2
):
    r"""
    :param psi_n_tar:
    :param sigma_n:
    :param rho_n:
    :param eta_n:
    :param eigen_fun_mat_tar:
    :param eigen_val_tar:
    :param source_data_n:
    :param E:
    :param signal_length_2:
    :return:
    """

    # initialization
    Lambda_n_tar_temp = np.diag(eigen_val_tar)

    source_data_n_tar = source_data_n['target']
    size_n_tar = source_data_n_tar.shape[0]
    x_mat_n_tar_sum = np.sum(source_data_n_tar, axis=0).T

    corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
    corr_n_t_inv = compute_inverse_matrix(corr_n_t)

    Lambda_n_tar_temp = Lambda_n_tar_temp + psi_n_tar ** 2 * size_n_tar * \
                         eigen_fun_mat_tar.T @ corr_n_t_inv @ eigen_fun_mat_tar
    U_n_tar_temp = psi_n_tar * eigen_fun_mat_tar.T @ corr_n_t_inv @ x_mat_n_tar_sum

    Lambda_n_tar = compute_inverse_matrix(Lambda_n_tar_temp)
    U_n_tar = (Lambda_n_tar @ U_n_tar_temp).T
    cov_n_s = create_cs_cov_mat(sigma_n ** 2, eta_n, E)

    A_n_tar_matn_obj = create_mat_normal_obj(U_n_tar, cov_n_s, Lambda_n_tar)
    A_n_tar_sample = A_n_tar_matn_obj.rvs(size=1)

    return A_n_tar_sample


def update_A_0_ntar_multi(
        psi_0_ntar, sigma, rho, eta,
        eigen_fun_mat_ntar, eigen_val_ntar,
        new_data, E, signal_length_2
):
    r"""
    :param psi_0_ntar:
    :param sigma:
    :param rho:
    :param eta:
    :param eigen_fun_mat_ntar:
    :param eigen_val_ntar:
    :param new_data:
    :param E:
    :param signal_length_2:
    :return: a list of two arrays, each in the format of vectorization.
    """

    # initialization
    eigen_val_ntar_len = len(eigen_val_ntar)
    Lambda_0_ntar_temp = np.diag(eigen_val_ntar)
    U_0_ntar_temp = np.zeros([eigen_val_ntar_len, E])

    sigma_0 = sigma[0, :]
    rho_0 = rho[0]
    eta_0 = eta[0]

    size_0_ntar = 0
    x_mat_0_ntar_sum = np.zeros([signal_length_2, E])

    new_data_ntar = new_data['non-target']
    size_0_ntar = size_0_ntar + new_data_ntar.shape[0]
    x_mat_0_ntar_sum = x_mat_0_ntar_sum + np.sum(new_data_ntar, axis=0).T

    corr_0_t = create_exponential_decay_cov_mat(1.0, rho_0, signal_length_2)
    corr_0_t_inv = compute_inverse_matrix(corr_0_t)

    Lambda_0_ntar_temp = Lambda_0_ntar_temp + psi_0_ntar ** 2 * size_0_ntar * \
                         eigen_fun_mat_ntar.T @ corr_0_t_inv @ eigen_fun_mat_ntar
    U_0_ntar_temp = U_0_ntar_temp + psi_0_ntar * eigen_fun_mat_ntar.T @ corr_0_t_inv @ x_mat_0_ntar_sum

    Lambda_0_ntar = compute_inverse_matrix(Lambda_0_ntar_temp)
    U_0_ntar = (Lambda_0_ntar @ U_0_ntar_temp).T
    cov_0_s = create_cs_cov_mat(sigma_0 ** 2, eta_0, E)

    A_0_ntar_matn_obj = create_mat_normal_obj(U_0_ntar, cov_0_s, Lambda_0_ntar)
    A_0_ntar_sample = A_0_ntar_matn_obj.rvs(size=1)

    return A_0_ntar_sample


def update_A_0_ntar_multi_alternative(
        scenario_dir, E, seq_size, xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    r"""
    :param scenario_dir:
    :param E: integer
    :param seq_size:
    :param source_name_ls:
    :param xdawn_bool:
    :param letter_dim_sub:
    :param kwargs:
    :return: For A_0_ntar, no need to re-perform MCMC; simply recycle an MCMC sample from reference method output.
    """

    ref_mcmc_id = kwargs['ref_mcmc_id']
    mcmc_output_dict = select_ref_mcmc_output_dict_alternative(source_name_ls, letter_dim_sub, xdawn_bool, 0, seq_size,
                                                               scenario_dir, **kwargs)
    A_0_ntar_sample = []
    if ref_mcmc_id is None:
        for e in range(E):
            A_0_ntar_sample.append(np.mean(mcmc_output_dict['alpha_ntar_0_{}'.format(e)], axis=0))
    else:
        for e in range(E):
            A_0_ntar_sample.append(mcmc_output_dict['alpha_ntar_0_{}'.format(e)][ref_mcmc_id, :])
    A_0_ntar_sample = np.stack(A_0_ntar_sample, axis=0)

    return A_0_ntar_sample


def update_psi_n_tar_multi_RWMH(
        A_tar, psi_tar_old, sigma, rho, eta, z_vector,
        eigen_fun_mat_tar, source_data, new_data, N_total, E, n, signal_length_2,
        step_size, scenario_dir, seq_source_size_2,
        xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    accept_n_bool = 0  # the acceptance bool is n-specific, which is a scalar!
    ln_loc = kwargs['psi_loc']
    ln_scale = kwargs['psi_scale']

    if n == 0:
        A_n_tar = A_tar[n, ...]
        psi_n_tar_old = psi_tar_old[n]
        sigma_n = sigma[n, :]
        rho_n = rho[n]
        eta_n = eta[n]

        corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
        cov_n_s = create_cs_cov_mat(sigma_n ** 2, eta_n, E)
        psi_n_tar_try = np.random.normal(loc=psi_n_tar_old, scale=step_size, size=None)

        if psi_n_tar_try > 0:
            # prior log-likelihood (use log-normal (0, 1))
            prior_log_prob_n_tar_try = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_try)
            prior_log_prob_n_tar_old = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_old)

            # log-data-likelihood part
            data_log_prob_n_tar_try = 0
            data_log_prob_n_tar_old = 0

            # different means
            B_n_tar_try = psi_n_tar_try * A_n_tar @ eigen_fun_mat_tar.T
            B_n_tar_old = psi_n_tar_old * A_n_tar @ eigen_fun_mat_tar.T

            matn_n_tar_try = create_mat_normal_obj(B_n_tar_try, cov_n_s, corr_n_t)
            matn_n_tar_old = create_mat_normal_obj(B_n_tar_old, cov_n_s, corr_n_t)

            # start from new data
            new_data_tar = new_data['target']
            data_log_prob_n_tar_try = data_log_prob_n_tar_try + np.sum(matn_n_tar_try.logpdf(new_data_tar))
            data_log_prob_n_tar_old = data_log_prob_n_tar_old + np.sum(matn_n_tar_old.logpdf(new_data_tar))

            # loop through source participant pool
            for nn in range(N_total - 1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_nn_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_nn_name = source_name_ls[nn + 1]
                    source_data_nn_tar = source_data[subject_nn_name]['target']
                    data_log_prob_n_tar_try = data_log_prob_n_tar_try + \
                                              np.sum(matn_n_tar_try.logpdf(source_data_nn_tar))
                    data_log_prob_n_tar_old = data_log_prob_n_tar_old + \
                                              np.sum(matn_n_tar_old.logpdf(source_data_nn_tar))

            alpha_log_ratio = data_log_prob_n_tar_try + prior_log_prob_n_tar_try - \
                              data_log_prob_n_tar_old - prior_log_prob_n_tar_old

            alpha_log_ratio = np.min([0, alpha_log_ratio])
            if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
                psi_n_tar_old = psi_n_tar_try
                accept_n_bool = 1
    else:
        ref_mcmc_bool = kwargs['ref_mcmc_bool']
        ref_mcmc_id = kwargs['ref_mcmc_id']

        if ref_mcmc_bool:
            mcmc_output_dict = select_proper_mcmc_output_dict(
                source_name_ls, scenario_dir, xdawn_bool, n,
                seq_source_size_2, letter_dim_sub, **kwargs
            )
            if ref_mcmc_id is None:
                psi_n_tar_old = np.mean(mcmc_output_dict['psi_tar_0'])
            else:
                psi_n_tar_old = mcmc_output_dict['psi_tar_0'][0, ref_mcmc_id]
        else:
            if z_vector[n-1] == 1:
                psi_n_tar_old = np.copy(psi_tar_old[0])  # assume psi_n_tar = psi_0_tar, n>0 if z_n=1
            else:
                if source_name_ls is None:
                    subject_n_name = 'subject_{}'.format(n)
                else:
                    subject_n_name = source_name_ls[n]

                psi_n_tar_old, accept_n_bool = update_psi_n_tar_ref_multi_RWMH(
                    A_tar[n, :], psi_tar_old[n], sigma[n], rho[n], eta[n],
                    eigen_fun_mat_tar, source_data[subject_n_name], E, signal_length_2,
                    step_size, **kwargs
                )

    psi_tar_old[n] = np.copy(psi_n_tar_old)

    return psi_tar_old, accept_n_bool


def update_psi_n_tar_ref_multi_RWMH(
        A_n_tar, psi_n_tar_old, sigma_n, rho_n, eta_n,
        eigen_fun_mat_tar, source_data, E, signal_length_2,
        step_size, **kwargs
):
    accept_n_bool = 0
    ln_loc = kwargs['psi_loc']
    ln_scale = kwargs['psi_scale']

    corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
    cov_n_s = create_cs_cov_mat(sigma_n ** 2, eta_n, E)
    psi_n_tar_try = np.random.normal(loc=psi_n_tar_old, scale=step_size, size=1)

    if psi_n_tar_try > 0:
        # prior log-likelihood (use log-normal (0, 1))
        prior_log_prob_n_tar_try = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_try)
        prior_log_prob_n_tar_old = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_old)

        # log-data-likelihood part
        data_log_prob_n_tar_try = 0
        data_log_prob_n_tar_old = 0

        # different means
        B_n_tar_try = psi_n_tar_try * A_n_tar @ eigen_fun_mat_tar.T
        B_n_tar_old = psi_n_tar_old * A_n_tar @ eigen_fun_mat_tar.T

        matn_n_tar_try = create_mat_normal_obj(B_n_tar_try, cov_n_s, corr_n_t)
        matn_n_tar_old = create_mat_normal_obj(B_n_tar_old, cov_n_s, corr_n_t)

        source_data_n_tar = source_data['non-target']
        data_log_prob_n_tar_try = data_log_prob_n_tar_try + \
                                   np.sum(matn_n_tar_try.logpdf(source_data_n_tar))
        data_log_prob_n_tar_old = data_log_prob_n_tar_old + \
                                   np.sum(matn_n_tar_old.logpdf(source_data_n_tar))

        alpha_log_ratio = data_log_prob_n_tar_try + prior_log_prob_n_tar_try - \
                          data_log_prob_n_tar_old - prior_log_prob_n_tar_old
        alpha_log_ratio = np.min([0, alpha_log_ratio[0]])

        if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
            psi_n_tar_old = psi_n_tar_try
            accept_n_bool = 1

    return psi_n_tar_old, accept_n_bool


def update_psi_0_ntar_multi_RWMH(
        A_0_ntar, psi_0_ntar_old, sigma, rho, eta,
        eigen_fun_mat_ntar, new_data, E, signal_length_2,
        step_size, **kwargs
):
    accept_bool = 0
    ln_loc = kwargs['psi_loc']
    ln_scale = kwargs['psi_scale']

    sigma_0 = sigma[0, :]
    rho_0 = rho[0]
    eta_0 = eta[0]

    corr_0_t = create_exponential_decay_cov_mat(1.0, rho_0, signal_length_2)
    cov_0_s = create_cs_cov_mat(sigma_0 ** 2, eta_0, E)
    psi_0_ntar_try = np.random.normal(loc=psi_0_ntar_old, scale=step_size, size=1)

    if psi_0_ntar_try > 0:
        # prior log-likelihood (use log-normal (0, 1))
        prior_log_prob_0_ntar_try = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_0_ntar_try)
        prior_log_prob_0_ntar_old = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_0_ntar_old)

        # log-data-likelihood part
        data_log_prob_0_ntar_try = 0
        data_log_prob_0_ntar_old = 0

        # different means
        B_0_ntar_try = psi_0_ntar_try * A_0_ntar @ eigen_fun_mat_ntar.T
        B_0_ntar_old = psi_0_ntar_old * A_0_ntar @ eigen_fun_mat_ntar.T

        matn_0_ntar_try = create_mat_normal_obj(B_0_ntar_try, cov_0_s, corr_0_t)
        matn_0_ntar_old = create_mat_normal_obj(B_0_ntar_old, cov_0_s, corr_0_t)

        new_data_ntar = new_data['non-target']
        data_log_prob_0_ntar_try = data_log_prob_0_ntar_try + \
                                   np.sum(matn_0_ntar_try.logpdf(new_data_ntar))
        data_log_prob_0_ntar_old = data_log_prob_0_ntar_old + \
                                   np.sum(matn_0_ntar_old.logpdf(new_data_ntar))

        alpha_log_ratio = data_log_prob_0_ntar_try + prior_log_prob_0_ntar_try - \
                          data_log_prob_0_ntar_old - prior_log_prob_0_ntar_old
        alpha_log_ratio = np.min([0, alpha_log_ratio[0]])

        if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
            psi_0_ntar_old = psi_0_ntar_try
            accept_bool = 1

    return psi_0_ntar_old, accept_bool


def update_psi_0_ntar_multi_alternative(
        scenario_dir, seq_size, xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    ref_mcmc_id = kwargs['ref_mcmc_id']
    mcmc_output_dict = select_ref_mcmc_output_dict_alternative(
        source_name_ls, letter_dim_sub, xdawn_bool, 0, seq_size,
        scenario_dir, **kwargs
    )
    if ref_mcmc_id is None:
        psi_0_ntar_sample = np.mean(mcmc_output_dict['psi_ntar_0'])
    else:
        psi_0_ntar_sample = mcmc_output_dict['psi_ntar_0'][0, ref_mcmc_id]

    return psi_0_ntar_sample


def update_sigma_n_multi_RWMH(
        B_tar, B_0_ntar, sigma_old, rho, eta, z_vector,
        source_data, new_data, N_total, E, n, signal_length_2,
        scenario_dir, seq_source_size_2,
        step_size, xdawn_bool, letter_dim_sub,
        source_name_ls=None, eigen_fun_mat_dict=None, **kwargs
):
    accept_n_bool = np.zeros([E])
    sigma_loc = kwargs['sigma_loc']
    sigma_scale = kwargs['sigma_scale']

    if n == 0:
        B_n_tar = B_tar[n, ...]
        rho_n = rho[n]
        eta_n = eta[n]
        corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)

        for e in range(E):
            sigma_n_try = np.copy(sigma_old[n, :])
            # use random walk as the proposal distribution
            sigma_n_e_try = np.random.normal(loc=sigma_old[n, e], scale=step_size[e], size=None)

            if sigma_n_e_try > 0.1:
                sigma_n_try[e] = sigma_n_e_try
                '''
                (use half-cauchy (0, 1))
                prior_log_prob_n_e_try = halfcauchy(loc=hc_loc, scale=hc_scale).logpdf(x=sigma_n_e_try)
                prior_log_prob_n_e_old = halfcauchy(loc=hc_loc, scale=hc_scale).logpdf(x=sigma_old[n, e])
                '''
                # Try inverse-gamma
                prior_inverse_gamma_obj = invgamma(a=sigma_loc, loc=0, scale=sigma_scale)
                prior_log_prob_n_e_try = prior_inverse_gamma_obj.logpdf(x=sigma_n_e_try)
                prior_log_prob_n_e_old = prior_inverse_gamma_obj.logpdf(x=sigma_old[n, e])

                # log-data-likelihood part
                data_log_prob_n_e_try = 0
                data_log_prob_n_e_old = 0

                # different covariance matrices
                cov_n_s_e_try = create_cs_cov_mat(sigma_n_try ** 2, eta_n, E)
                cov_n_s_e_old = create_cs_cov_mat(sigma_old[n, :] ** 2, eta_n, E)

                matn_n_tar_e_try = create_mat_normal_obj(B_n_tar, cov_n_s_e_try, corr_n_t)
                matn_n_tar_e_old = create_mat_normal_obj(B_n_tar, cov_n_s_e_old, corr_n_t)

                matn_n_ntar_e_try = create_mat_normal_obj(B_0_ntar, cov_n_s_e_try, corr_n_t)
                matn_n_ntar_e_old = create_mat_normal_obj(B_0_ntar, cov_n_s_e_old, corr_n_t)

                # new data
                new_data_tar = new_data['target']
                new_data_ntar = new_data['non-target']

                data_log_prob_n_e_try = data_log_prob_n_e_try + \
                                        np.sum(matn_n_tar_e_try.logpdf(X=new_data_tar)) + \
                                        np.sum(matn_n_ntar_e_try.logpdf(X=new_data_ntar))
                data_log_prob_n_e_old = data_log_prob_n_e_old + \
                                        np.sum(matn_n_tar_e_old.logpdf(X=new_data_tar)) + \
                                        np.sum(matn_n_ntar_e_old.logpdf(X=new_data_ntar))

                # source data
                for nn in range(N_total - 1):
                    if z_vector[nn] == 1:
                        if source_name_ls is None:
                            subject_nn_name = 'subject_{}'.format(nn + 1)
                        else:
                            subject_nn_name = source_name_ls[nn + 1]
                        source_data_nn_tar = source_data[subject_nn_name]['target']
                        data_log_prob_n_e_try = data_log_prob_n_e_try + \
                                                np.sum(matn_n_tar_e_try.logpdf(X=source_data_nn_tar))
                        data_log_prob_n_e_old = data_log_prob_n_e_old + \
                                                np.sum(matn_n_tar_e_old.logpdf(X=source_data_nn_tar))

                alpha_log_n_e_ratio = data_log_prob_n_e_try + prior_log_prob_n_e_try - \
                                      data_log_prob_n_e_old - prior_log_prob_n_e_old

                alpha_log_n_e_ratio = np.min([0, alpha_log_n_e_ratio])
                if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_n_e_ratio):
                    sigma_old[n, e] = np.copy(sigma_n_e_try)
                    accept_n_bool[e] = 1
    else:
        ref_mcmc_bool = kwargs['ref_mcmc_bool']
        ref_mcmc_id = kwargs['ref_mcmc_id']

        if ref_mcmc_bool:
            # use source data's own model fit
            mcmc_output_dict = select_ref_mcmc_output_dict_alternative(
                source_name_ls, letter_dim_sub, xdawn_bool, n,
                seq_source_size_2, scenario_dir, **kwargs
            )
            sigma_n_sample = []
            if ref_mcmc_id is None:
                for e in range(E):
                    sigma_n_sample.append(np.mean(mcmc_output_dict['sigma_0_{}'.format(e)]))
            else:
                for e in range(E):
                    sigma_n_sample.append(mcmc_output_dict['sigma_0_{}'.format(e)][0, ref_mcmc_id])
            sigma_n_sample = np.stack(sigma_n_sample, axis=0)
            sigma_old[n, :] = np.copy(sigma_n_sample)
        else:
            if z_vector[n - 1] == 1:
                sigma_old[n, :] = np.copy(sigma_old[0, :])
                accept_n_bool = np.ones([E])
            else:
                if source_name_ls is None:
                    subject_n_name = 'subject_{}'.format(n)
                else:
                    subject_n_name = source_name_ls[n]
                sigma_old, accept_n_bool = update_sigma_source_n_tar_multi_IndMH(
                    B_tar[n, :], sigma_old, rho[n], eta[n],
                    source_data[subject_n_name], E, n, signal_length_2,
                    **kwargs
                )

    return sigma_old, accept_n_bool


def update_sigma_source_n_tar_multi_IndMH(
        B_n_tar, sigma_old, rho_n, eta_n,
        source_data_n, E, n, signal_length_2,
        **kwargs
):
    accept_n_bool = np.zeros([E])
    inv_gamma_a = kwargs['sigma_loc']
    inv_gamma_b = kwargs['sigma_scale']

    corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
    # sigma_0_ref = np.copy(sigma_n_old[0, :])

    # source data with target signals only
    source_data_n_tar = source_data_n['target']
    # source_data_n_ntar = source_data[source_name_ls[n]]['non-target']

    for e in range(E):
        sigma_n_try = np.copy(sigma_old[n, :])
        # use exp(1) distribution as proposal
        sigma_n_e_try = np.random.exponential(scale=1.0, size=None)

        if sigma_n_e_try > 0.1:
            sigma_n_try[e] = sigma_n_e_try

            # log proposal ratio
            q_log_prob_n_e_try = expon(loc=0.0, scale=1.0).logpdf(sigma_n_e_try)
            q_log_prob_n_e_old = expon(loc=0.0, scale=1.0).logpdf(sigma_old[n, e])

            # Try inverse-gamma prior
            inv_gamma_rv_obj = invgamma(a=inv_gamma_a, loc=0, scale=inv_gamma_b)
            prior_log_prob_n_e_try = inv_gamma_rv_obj.logpdf(x=sigma_n_e_try)
            prior_log_prob_n_e_old = inv_gamma_rv_obj.logpdf(x=sigma_old[n, e])

            # different covariance matrices
            cov_n_s_e_try = create_cs_cov_mat(sigma_n_try ** 2, eta_n, E)
            cov_n_s_e_old = create_cs_cov_mat(sigma_old[n, :] ** 2, eta_n, E)

            matn_n_tar_e_try = create_mat_normal_obj(B_n_tar, cov_n_s_e_try, corr_n_t)
            matn_n_tar_e_old = create_mat_normal_obj(B_n_tar, cov_n_s_e_old, corr_n_t)

            # matn_n_ntar_e_try = create_mat_normal_obj(B_n_ntar, cov_n_s_e_try, corr_n_t)
            # matn_n_ntar_e_old = create_mat_normal_obj(B_n_ntar, cov_n_s_e_old, corr_n_t)

            # log-data-likelihood part (source data n)
            data_log_prob_n_e_try = np.sum(matn_n_tar_e_try.logpdf(X=source_data_n_tar))
                                    # np.sum(matn_n_ntar_e_try.logpdf(X=source_data_n_ntar))
            data_log_prob_n_e_old = np.sum(matn_n_tar_e_old.logpdf(X=source_data_n_tar))
                                    # np.sum(matn_n_ntar_e_old.logpdf(X=source_data_n_ntar))

            alpha_log_n_e_ratio = prior_log_prob_n_e_try - prior_log_prob_n_e_old + \
                                  data_log_prob_n_e_try - data_log_prob_n_e_old + \
                                  q_log_prob_n_e_old - q_log_prob_n_e_try

            alpha_log_n_e_ratio = np.min([0, alpha_log_n_e_ratio])
            if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_n_e_ratio):
                sigma_old[n, e] = np.copy(sigma_n_e_try)
                accept_n_bool[e] = 1

    return sigma_old, accept_n_bool


def update_eta_n_multi_discrete_uniform(
        B_tar, B_0_ntar, sigma, rho, eta_old, z_vector,
        source_data, new_data, N_total, E, n, signal_length_2,
        scenario_dir, seq_source_size_2,
        xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    eta_grid = kwargs['eta_grid']

    if n == 0:
        B_n_tar = B_tar[n, ...]
        sigma_n = sigma[n, :]
        rho_n = rho[n]
        corr_n_t = create_exponential_decay_cov_mat(1.0, rho_n, signal_length_2)
        data_log_prob_n = []

        for eta_n_iter in eta_grid:
            cov_n_s_iter = create_cs_cov_mat(sigma_n ** 2, eta_n_iter, E)
            matn_n_tar_iter = create_mat_normal_obj(B_n_tar, cov_n_s_iter, corr_n_t)
            matn_n_ntar_iter = create_mat_normal_obj(B_0_ntar, cov_n_s_iter, corr_n_t)

            new_data_tar = new_data['target']
            new_data_ntar = new_data['non-target']
            data_log_prob_n_iter = np.sum(matn_n_tar_iter.logpdf(new_data_tar)) + \
                                   np.sum(matn_n_ntar_iter.logpdf(new_data_ntar))

            for nn in range(N_total - 1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_n_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_n_name = source_name_ls[nn + 1]
                    source_data_n_tar = source_data[subject_n_name]['target']
                    data_log_prob_n_iter = data_log_prob_n_iter + \
                                           np.sum(matn_n_tar_iter.logpdf(X=source_data_n_tar))

            data_log_prob_n.append(data_log_prob_n_iter)

        data_log_prob_n = np.stack(data_log_prob_n, axis=0)
        data_prob_n = convert_log_prob_to_prob(data_log_prob_n)
        eta_n_sample = eta_grid[np.where(mtn(n=1, p=data_prob_n).rvs(size=None) == 1)[0][0]]
        eta_old[n] = np.copy(eta_n_sample)

    else:
        ref_mcmc_id = kwargs['ref_mcmc_id']
        mcmc_output_dict = select_ref_mcmc_output_dict_alternative(source_name_ls, letter_dim_sub, xdawn_bool, n,
                                                                   seq_source_size_2, scenario_dir, **kwargs)
        if ref_mcmc_id is None:
            eta_n_sample = find_nearby_parameter_grid(np.mean(np.squeeze(mcmc_output_dict['eta_0'])), eta_grid)
        else:
            eta_n_sample = find_nearby_parameter_grid(np.squeeze(mcmc_output_dict['eta_0'])[ref_mcmc_id], eta_grid)
        eta_old[n] = np.copy(eta_n_sample)

    return eta_old


def update_rho_n_multi_discrete_uniform(
        B_tar, B_0_ntar, sigma, rho_old, eta, z_vector,
        source_data, new_data, N_total, E, n, signal_length_2,
        scenario_dir, seq_source_size_2,
        xdawn_bool, letter_dim_sub,
        source_name_ls=None, **kwargs
):
    rho_grid = kwargs['rho_grid']
    ref_mcmc_id = kwargs['ref_mcmc_id']

    if n == 0:
        B_n_tar = B_tar[n, ...]
        sigma_n = sigma[n, :]
        eta_n = eta[n]
        cov_n_s = create_cs_cov_mat(sigma_n ** 2, eta_n, E)
        data_log_prob_n = []

        for rho_n_iter in rho_grid:
            corr_n_t_iter = create_exponential_decay_cov_mat(1.0, rho_n_iter, signal_length_2)
            matn_n_tar_iter = create_mat_normal_obj(B_n_tar, cov_n_s, corr_n_t_iter)
            matn_n_ntar_iter = create_mat_normal_obj(B_0_ntar, cov_n_s, corr_n_t_iter)

            new_data_tar = new_data['target']
            new_data_ntar = new_data['non-target']
            data_log_prob_n_iter = np.sum(matn_n_tar_iter.logpdf(new_data_tar)) + \
                                   np.sum(matn_n_ntar_iter.logpdf(new_data_ntar))

            for nn in range(N_total - 1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_n_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_n_name = source_name_ls[nn + 1]
                    source_data_n_tar = source_data[subject_n_name]['target']
                    data_log_prob_n_iter = data_log_prob_n_iter + \
                                           np.sum(matn_n_tar_iter.logpdf(X=source_data_n_tar))
            data_log_prob_n.append(data_log_prob_n_iter)

        data_log_prob_n = np.stack(data_log_prob_n, axis=0)
        data_prob_n = convert_log_prob_to_prob(data_log_prob_n)
        rho_n_sample = rho_grid[np.where(mtn(n=1, p=data_prob_n).rvs(size=None) == 1)[0][0]]
        # rho_old[n] = np.copy(rho_n_sample)

    else:
        mcmc_output_dict = select_ref_mcmc_output_dict_alternative(source_name_ls, letter_dim_sub, xdawn_bool, n,
                                                                   seq_source_size_2, scenario_dir, **kwargs)
        if ref_mcmc_id is None:
            rho_n_sample = find_nearby_parameter_grid(np.mean(np.squeeze(mcmc_output_dict['rho_0'])), rho_grid)
        else:
            rho_n_sample = find_nearby_parameter_grid(np.squeeze(mcmc_output_dict['rho_0'])[ref_mcmc_id], rho_grid)

    rho_old[n] = np.copy(rho_n_sample)

    return rho_old


def update_Z_indicator_n_multi(
        B_tar_0, B_tar_n, sigma_0, sigma_n, rho_0, rho_n, eta_0, eta_n,
        source_data_n, E, signal_length_2, **kwargs
):
    approx_threshold = kwargs['approx_threshold']
    approx_random_bool = kwargs['approx_random_bool']
    # if approx_random_bool is True, we reset the normalized probability to be [0.5, 0.5],
    # otherwise, we will reset the normalized probability to be [0.99, 0.01] and
    # it is almost sure that we will obtain a sample of 1.
    source_data_n_tar = source_data_n['target']
    z_n_match_log_prob = kwargs['z_n_match_prior_prob']

    matn_0_tar = create_mat_normal_obj_from_mcmc(
        B_tar_0, rho_0, eta_0, sigma_0, signal_length_2, E
    )
    data_log_prob_0 = np.sum(matn_0_tar.logpdf(X=source_data_n_tar))

    matn_n_tar = create_mat_normal_obj_from_mcmc(
        B_tar_n, rho_n, eta_n, sigma_n, signal_length_2, E
    )
    data_log_prob_n = np.sum(matn_n_tar.logpdf(X=source_data_n_tar))
    data_log_prob_vec = np.array([data_log_prob_0 + np.log(z_n_match_log_prob),
                                  data_log_prob_n + np.log(1-z_n_match_log_prob)])
    log_prob_diff_abs = np.abs(data_log_prob_vec[1] - data_log_prob_vec[0])

    source_data_n_tar_size = source_data_n_tar.shape[0]
    # source_data_n_tar_size = 1
    if log_prob_diff_abs / source_data_n_tar_size < approx_threshold:
        if approx_random_bool:
            normalized_prob_n = np.array([0.5, 0.5])
        else:
            normalized_prob_n = np.array([0.99, 0.01])
    else:
        normalized_prob_n = convert_log_prob_to_prob(data_log_prob_vec)

    sample_n = np.random.binomial(1, normalized_prob_n[0], size=None)
    return data_log_prob_vec, sample_n


def update_Z_indicator_multi(
        B_tar_iter, sigma_iter, rho_iter, eta_iter, N_total, E, feature_vec_len,
        source_name_ls, source_data, **kwargs
):
    z_vector_iter = []
    log_prob_vec_iter = []
    # initialize parameter associated with cluster 0
    B_0_tar_iter = np.copy(B_tar_iter[0, ...])
    sigma_0_iter = np.copy(sigma_iter[0, :])
    rho_0_iter = np.copy(rho_iter[0])
    eta_0_iter = np.copy(eta_iter[0])

    for nn in range(N_total - 1):
        if source_name_ls is None:
            name_n = 'subject_{}'.format(nn + 1)
        else:
            name_n = source_name_ls[nn + 1]
        source_data_n = source_data[name_n]

        B_n_tar_ref_iter = B_tar_iter[nn + 1, ...]
        sigma_n_ref_iter = sigma_iter[nn + 1, :]
        rho_n_ref_iter = rho_iter[nn + 1]
        eta_n_ref_iter = eta_iter[nn + 1]

        log_prob_n, z_vector_iter_n = update_Z_indicator_n_multi(B_0_tar_iter, B_n_tar_ref_iter, sigma_0_iter,
                                                                 sigma_n_ref_iter, rho_0_iter, rho_n_ref_iter,
                                                                 eta_0_iter, eta_n_ref_iter, source_data_n, E,
                                                                 feature_vec_len, **kwargs)
        z_vector_iter.append(z_vector_iter_n)
        log_prob_vec_iter.append(log_prob_n)
    z_vector_iter = np.array(z_vector_iter)
    log_prob_vec_iter = np.array(log_prob_vec_iter)
    # print('log_prob_vec_iter = {}'.format(log_prob_vec_iter))
    return z_vector_iter


def compute_joint_log_likelihood_cluster_multi(
        B_tar_iter, B_0_ntar_iter, A_tar_iter, A_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter,
        sigma_iter, rho_iter, eta_iter, z_vector_iter,
        eigen_val_dict, source_data, new_data, N_total, E, signal_length_2,
        source_name_ls=None, **kwargs
):
    # compute the joint log-likelihood
    log_prior_N_plus = 0
    log_source_data = 0
    log_new_data = 0

    psi_loc = kwargs['psi_loc']
    psi_scale = kwargs['psi_scale']
    sigma_loc = kwargs['sigma_loc']
    sigma_scale = kwargs['sigma_scale']

    matn_N_plus_tar_obj = []
    matn_0_ntar_obj = []
    invgamma_obj = invgamma(a=sigma_loc, loc=0, scale=sigma_scale)

    # prior part
    for n in range(N_total):

        eigen_val_tar = eigen_val_dict['target']
        alpha_tar_size = len(eigen_val_tar)

        cov_n_s = create_cs_cov_mat(sigma_iter[n, :] ** 2, eta_iter[n], E)
        corr_t_A_n_tar = np.diag(eigen_val_tar)
        corr_n_t = create_exponential_decay_cov_mat(1.0, rho_iter[n], signal_length_2)

        A_n_tar_obj = create_mat_normal_obj(np.zeros([E, alpha_tar_size]), cov_n_s, corr_t_A_n_tar)
        log_prior_A_n_tar = A_n_tar_obj.logpdf(X=A_tar_iter[n, ...])

        log_prior_psi_n_tar = np.sum(lognorm(loc=psi_loc, s=psi_scale).logpdf(x=psi_tar_iter[n]))
        log_prior_sigma_n = 0

        for xdawn_e in range(E):
            log_prior_sigma_n = log_prior_sigma_n + \
                                np.sum(invgamma_obj.logpdf(x=sigma_iter[n, xdawn_e]))
        # discrete uniform, no log prior contributions from rho nor eta.
        # log_prior_rho_n = beta(a=rho_alpha, b=rho_beta).logpdf(x=rho_iter[n])
        # log_prior_eta_n = beta(a=eta_alpha, b=eta_beta).logpdf(x=eta_iter[n])

        log_prior_N_plus = log_prior_N_plus + log_prior_A_n_tar + log_prior_psi_n_tar + log_prior_sigma_n

        matn_n_tar_obj = create_mat_normal_obj(B_tar_iter[n, ...], cov_n_s, corr_n_t)
        matn_N_plus_tar_obj.append(matn_n_tar_obj)

        if n == 0:
            eigen_val_ntar = eigen_val_dict['non-target']
            alpha_ntar_size = len(eigen_val_ntar)
            corr_t_A_0_ntar = np.diag(eigen_val_ntar)
            A_n_ntar_obj = create_mat_normal_obj(np.zeros([E, alpha_ntar_size]), cov_n_s, corr_t_A_0_ntar)

            log_prior_A_0_ntar = A_n_ntar_obj.logpdf(X=A_0_ntar_iter)
            log_prior_psi_0_ntar = np.sum(lognorm(loc=psi_loc, s=psi_scale).logpdf(x=psi_0_ntar_iter))

            log_prior_N_plus = log_prior_N_plus + log_prior_A_0_ntar + log_prior_psi_0_ntar
            matn_0_ntar_obj.append(matn(B_0_ntar_iter, cov_n_s, corr_n_t))

    # source data part
    for nn in range(N_total - 1):
        if source_name_ls is None:
            subject_n_name = 'subject_{}'.format(nn + 1)
        else:
            subject_n_name = source_name_ls[nn + 1]

        source_data_n_tar = source_data[subject_n_name]['target']
        if z_vector_iter[nn] == 1:
            log_data_n_tar = np.sum(matn_N_plus_tar_obj[0].logpdf(X=source_data_n_tar))
            # log_n_indicator = np.log(z_vector_prob_iter[n])
        else:
            log_data_n_tar = np.sum(matn_N_plus_tar_obj[nn + 1].logpdf(X=source_data_n_tar))
            # log_n_indicator = -np.log(z_vector_prob_iter[n])
        log_source_data = log_source_data + log_data_n_tar

    # new data contribution
    new_data_tar = new_data['target']
    log_data_0_tar = np.sum(matn_N_plus_tar_obj[0].logpdf(X=new_data_tar))
    new_data_ntar = new_data['non-target']
    log_data_0_ntar = np.sum(matn_0_ntar_obj[0].logpdf(X=new_data_ntar))
    log_new_data = log_new_data + log_data_0_tar + log_data_0_ntar
    log_joint_prob = log_prior_N_plus + log_source_data + log_new_data

    return log_joint_prob
