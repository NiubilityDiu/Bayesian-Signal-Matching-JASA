from self_py_fun.Misc import *
from self_py_fun.SimFun import *
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax import random
import scipy.io as sio
from scipy.stats import matrix_normal as matn
from numpyro.infer import NUTS, MCMC


# multi-channel functions
# initialization and import-related
def is_matn_convertible(mean_coef, row_cov, col_cov):
    try:
        matn(mean=mean_coef, rowcov=row_cov, colcov=col_cov)
        return True
    except np.linalg.LinAlgError:
        print('matn has singular covariance matrix.')
        return False


def signal_new_sim_multi(
        input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None,
        E=None
):
    r"""
    :param input_points:
    :param eigen_val_dict:
    :param eigen_fun_mat_dict:
    :param input_data:
    :param E: int
    :return: This is a reference panel (restriction) for the new subject, try fitting GP separately
    """
    signal_length = len(input_points)
    beta_ntar_ls = []
    beta_tar_ls = []
    sigma_ls = []

    psi_ntar_name = 'psi_ntar_0'
    psi_ntar = numpyro.sample(psi_ntar_name, dist.LogNormal(loc=0.0, scale=1.0))
    psi_tar_name = 'psi_tar_0'
    psi_tar = numpyro.sample(psi_tar_name, dist.LogNormal(loc=0.0, scale=1.0))

    for e_id in range(E):
        alpha_ntar_e_name = 'alpha_ntar_0_{}'.format(e_id)
        alpha_ntar_e = numpyro.sample(
            alpha_ntar_e_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict['non-target']),
                covariance_matrix=jnp.diag(eigen_val_dict['non-target']))
        )
        alpha_tar_e_name = 'alpha_tar_0_{}'.format(e_id)
        alpha_tar_e = numpyro.sample(
            alpha_tar_e_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict['target']),
                covariance_matrix=jnp.diag(eigen_val_dict['target'])
            )
        )

        beta_ntar_e = psi_ntar * jnp.matmul(eigen_fun_mat_dict['non-target'], alpha_ntar_e)
        beta_tar_e = psi_tar * jnp.matmul(eigen_fun_mat_dict['target'], alpha_tar_e)

        sigma_e_name = 'sigma_0_{}'.format(e_id)
        sigma_e = numpyro.sample(sigma_e_name, dist.HalfCauchy(scale=5.0))

        beta_ntar_ls.append(beta_ntar_e)
        beta_tar_ls.append(beta_tar_e)
        sigma_ls.append(sigma_e)

    rho_name = 'rho_0'
    # rho = numpyro.sample(rho_name, dist.Beta(concentration0=1.0, concentration1=1.0))
    rho = numpyro.sample(rho_name, dist.Uniform(low=0.1, high=0.9))
    cov_0_temp_mat = create_exponential_decay_cov_mat_jnp(1.0, rho, signal_length)

    eta_name = 'eta_0'
    # eta = numpyro.sample(eta_name, dist.Beta(concentration0=1.0, concentration1=1.0))
    eta = numpyro.sample(eta_name, dist.Uniform(low=0.1, high=0.9))
    cov_0_spatial_mat = create_cs_cov_mat_jnp(sigma_ls, eta, E)

    data_0_ntar_name = 'subject_0_ntar'
    new_0_ntar_dist = dist.MultivariateNormal(
        loc=jnp.reshape(jnp.stack(beta_ntar_ls), [E * signal_length]),
        covariance_matrix=jnp.kron(cov_0_spatial_mat, cov_0_temp_mat)
    )
    numpyro.sample(data_0_ntar_name, new_0_ntar_dist, obs=input_data['non-target'])

    data_0_tar_name = 'subject_0_tar'
    new_0_tar_dist = dist.MultivariateNormal(
        loc=jnp.reshape(jnp.stack(beta_tar_ls), [E * signal_length]),
        covariance_matrix=jnp.kron(cov_0_spatial_mat, cov_0_temp_mat)
    )
    numpyro.sample(data_0_tar_name, new_0_tar_dist, obs=input_data['target'])


def import_mcmc_summary_reference_multi(reference_mcmc_dir, E, eigen_fun_mat_0_dict):
    reference_mcmc = sio.loadmat(reference_mcmc_dir)
    B_tar_0 = []
    B_ntar_0 = []
    A_tar_0 = []
    A_ntar_0 = []
    sigma_0 = []

    psi_tar_0 = reference_mcmc['psi_tar_0']
    psi_ntar_0 = reference_mcmc['psi_ntar_0']

    eigen_fun_mat_tar = eigen_fun_mat_0_dict['target']
    eigen_fun_mat_ntar = eigen_fun_mat_0_dict['non-target']

    for e in range(E):
        A_tar_0_e = reference_mcmc['alpha_tar_0_{}'.format(e)]
        A_tar_0.append(A_tar_0_e)
        B_tar_0_e = psi_tar_0.T * A_tar_0_e @ eigen_fun_mat_tar.T
        B_tar_0.append(B_tar_0_e)

        A_ntar_0_e = reference_mcmc['alpha_ntar_0_{}'.format(e)]
        A_ntar_0.append(A_ntar_0_e)
        B_ntar_0_e = psi_ntar_0.T * A_ntar_0_e @ eigen_fun_mat_ntar.T
        B_ntar_0.append(B_ntar_0_e)

        sigma_0_e = np.squeeze(reference_mcmc['sigma_0_{}'.format(e)], axis=0)
        sigma_0.append(sigma_0_e)

    # the dimension in the comment section refers to the operator inside np.mean().
    A_tar_0 = np.mean(np.stack(A_tar_0, axis=1), axis=0)  # (mcmc_num, E, feature_vector)
    A_ntar_0 = np.mean(np.stack(A_ntar_0, axis=1), axis=0)  # (mcmc_num, E, feature_vector)
    B_tar_0 = np.mean(np.stack(B_tar_0, axis=1), axis=0)  # (mcmc_num, E, signal_length)
    B_ntar_0 = np.mean(np.stack(B_ntar_0, axis=1), axis=0)  # (mcmc_num, E, signal_length)
    sigma_0 = np.mean(np.stack(sigma_0, axis=-1), axis=0)  # (mcmc_num, E)

    rho_0 = np.mean(reference_mcmc['rho_0'])
    eta_0 = np.mean(reference_mcmc['eta_0'])

    psi_tar_0 = np.mean(psi_tar_0)
    psi_ntar_0 = np.mean(psi_ntar_0)

    return B_tar_0, B_ntar_0, A_tar_0, A_ntar_0, psi_tar_0, psi_ntar_0, sigma_0, rho_0, eta_0


def initialize_data_multi_fast_calculation_sim(
        N_total, E, signal_length_2, seq_new_i, seq_source_size_2, new_data,
        eigen_fun_mat_dict, eigen_val_dict, scenario_name_dir
):

    A_tar_len = len(eigen_val_dict['target'])
    A_0_ntar_len = len(eigen_val_dict['non-target'])
    init_param_dict = {
        "B_tar": [np.ones(shape=[N_total, E, signal_length_2])],
        "B_0_ntar": [np.ones(shape=[E, signal_length_2])],
        "A_tar": [np.ones([N_total, E, A_tar_len])],
        "A_0_ntar": [np.ones([E, A_0_ntar_len])],
        "psi_tar": [np.ones(shape=N_total)],
        "psi_0_ntar": [1],
        "sigma": [np.ones([N_total, E])],
        "rho": [np.ones([N_total]) - 0.5],
        "eta": [np.ones([N_total]) - 0.5],
        "z_vector": [np.random.binomial(n=1, p=0.5, size=N_total-1)],
        # "z_vector_prob": [np.ones([N_total - 1]) - 0.5],
        "log_joint_prob": [0]
    }

    B_tar = []
    B_0_ntar = []
    A_tar = []
    A_0_ntar = []
    psi_tar = []
    psi_0_ntar = []
    sigma = []
    rho = []
    eta = []
    new_tar_log_prob = []

    for n in range(N_total):
        if n == 0:
            reference_n_dir = '{}/reference_numpyro/mcmc_sub_0_seq_size_{}_reference.mat'.format(
                scenario_name_dir, seq_new_i + 1
            )
        else:
            reference_n_dir = '{}/reference_numpyro/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                scenario_name_dir, n, seq_source_size_2
            )

        [B_tar_n, B_ntar_n, A_tar_n, A_ntar_n,
         psi_tar_n, psi_ntar_n, sigma_n, rho_n, eta_n] = import_mcmc_summary_reference_multi(
            reference_n_dir, E, eigen_fun_mat_dict
        )
        new_tar_log_prob_n, _ = _compute_new_data_multi_log_likelihood(
            E, signal_length_2, new_data, B_tar_n, B_ntar_n, sigma_n, eta_n, rho_n
        )

        B_tar.append(B_tar_n)
        A_tar.append(A_tar_n)
        psi_tar.append(psi_tar_n)
        sigma.append(sigma_n)
        rho.append(rho_n)
        eta.append(eta_n)
        new_tar_log_prob.append(new_tar_log_prob_n)

        if n == 0:
            B_0_ntar.append(B_ntar_n)
            A_0_ntar.append(A_ntar_n)
            psi_0_ntar.append(psi_ntar_n)

    B_tar = np.stack(B_tar, axis=0)
    A_tar = np.stack(A_tar, axis=0)
    psi_tar = np.stack(psi_tar, axis=0)
    sigma = np.stack(sigma, axis=0)
    rho = np.stack(rho, axis=0)
    eta = np.stack(eta, axis=0)
    new_tar_log_prob = np.array(new_tar_log_prob)

    B_0_ntar = np.squeeze(np.array(B_0_ntar), axis=0)
    A_0_ntar = np.squeeze(np.array(A_0_ntar), axis=0)
    psi_0_ntar = np.squeeze(np.array(psi_0_ntar), axis=0)

    z_vector_init = []
    # z_vector_prob_init = []

    for n in range(N_total - 1):
        z_vector_prob_n = convert_log_prob_to_prob(np.array([new_tar_log_prob[0], new_tar_log_prob[n + 1]]))[1]
        z_vector_init.append(np.random.binomial(1, z_vector_prob_n, size=None))
        # z_vector_prob_init.append(z_vector_prob_n)

    z_vector_init = np.stack(z_vector_init, axis=0)
    # z_vector_prob_init = np.stack(z_vector_prob_init, axis=0)

    init_param_dict['z_vector'][0] = np.copy(z_vector_init)
    # init_param_dict['z_vector_prob'][0] = np.copy(z_vector_prob_init)

    init_param_dict['B_tar'][0] = np.copy(B_tar)
    init_param_dict['B_0_ntar'][0] = np.copy(B_0_ntar)
    init_param_dict['A_tar'][0] = np.copy(A_tar)
    init_param_dict['A_0_ntar'][0] = np.copy(A_0_ntar)
    init_param_dict['psi_tar'][0] = np.copy(psi_tar)
    init_param_dict['psi_0_ntar'][0] = np.copy(psi_0_ntar)
    init_param_dict['sigma'][0] = np.copy(sigma)
    init_param_dict['eta'][0] = np.copy(eta)
    init_param_dict['rho'][0] = np.copy(rho)

    return init_param_dict


def determine_proper_ref_mcmc_output_dir(
        channel_ids_str, sub_new_name, sub_n, seq_new_i, seq_source_size_2, parent_dir,
        xdawn_bool, letter_dim_sub, **kwargs
):
    hyper_param_bool = kwargs['hyper_param_bool']
    sub_n_dir = '{}/{}'.format(parent_dir, sub_n)
    ref_method_name = 'reference_numpyro_letter_{}'.format(letter_dim_sub)
    channel_name = 'channel_{}'.format(channel_ids_str)
    if xdawn_bool:
        ref_method_name = '{}_xdawn'.format(ref_method_name)
        n_components = kwargs['n_components']
        channel_name = 'channel_all_comp_{}'.format(n_components)

    ref_method_dir = '{}/{}'.format(sub_n_dir, ref_method_name)
    ref_method_channel_dir = '{}/{}'.format(ref_method_dir, channel_name)

    if sub_n == sub_new_name:
        mcmc_file_name = 'mcmc_seq_size_{}_reference_xdawn.mat'.format(seq_new_i + 1)
    else:
        mcmc_file_name = 'mcmc_seq_size_{}_reference_xdawn.mat'.format(seq_source_size_2)

    if hyper_param_bool:
        kernel_name = kwargs['kernel_name']
        reference_n_dir = '{}/{}/{}'.format(
            ref_method_channel_dir, kernel_name, mcmc_file_name
        )
    else:
        reference_n_dir = '{}/{}'.format(
            ref_method_channel_dir, mcmc_file_name
        )

    return reference_n_dir


def initialize_data_multi_fast_calculation_eeg(
        E_num, xdawn_bool, letter_dim_sub,
        total_sub_name, seq_source_size_2, eigen_fun_dict, scenario_dir,
        **kwargs
):
    N_total = len(total_sub_name)
    ref_mcmc_id = 0
    z_vector_init = np.random.binomial(n=1, p=0.5, size=N_total - 1)

    init_param_dict = {
        "B_tar": [],
        "B_0_ntar": [],
        "A_tar": [],
        "A_0_ntar": [],
        "psi_tar": [],
        "psi_0_ntar": [],
        "sigma": [],
        "rho": [np.zeros([N_total]) + 0.5],
        "eta": [np.zeros([N_total])],
        "z_vector": [z_vector_init],
        "log_joint_prob": [0]
    }

    A_tar_iter = []
    B_tar_iter = []
    psi_tar_iter = []
    sigma_iter = []

    # Use reference values to start
    for n_iter in range(N_total):
        mcmc_output_dict_n_iter = select_ref_mcmc_output_dict_alternative(
            total_sub_name, letter_dim_sub, xdawn_bool, n_iter,
            seq_source_size_2, scenario_dir, **kwargs
        )
        # print(mcmc_output_dict_n_iter.keys())
        if n_iter > 0 and z_vector_init[n_iter-1] == 1:
            A_tar_n_iter = np.copy(A_tar_iter[0])
            B_tar_n_iter = np.copy(B_tar_iter[0])
            sigma_n_iter = np.copy(sigma_iter[0])
            psi_tar_n_iter = np.copy(psi_tar_iter[0])
        else:
            A_tar_n_iter = []
            B_tar_n_iter = []
            sigma_n_iter = []
            psi_tar_n_iter = mcmc_output_dict_n_iter['psi_tar_0'][0, ref_mcmc_id]

            for e_it in range(E_num):
                A_tar_n_e_iter = mcmc_output_dict_n_iter['alpha_tar_0_{}'.format(e_it)][ref_mcmc_id, :]
                A_tar_n_iter.append(A_tar_n_e_iter)
                B_tar_n_iter.append(eigen_fun_dict['target'] @ A_tar_n_e_iter)
                sigma_n_iter.append(mcmc_output_dict_n_iter['sigma_0_{}'.format(e_it)][0, ref_mcmc_id])

            A_tar_n_iter = np.stack(A_tar_n_iter, axis=0)
            B_tar_n_iter = psi_tar_n_iter * np.stack(B_tar_n_iter, axis=0)
            sigma_n_iter = np.stack(sigma_n_iter, axis=0)

        A_tar_iter.append(A_tar_n_iter)
        B_tar_iter.append(B_tar_n_iter)
        psi_tar_iter.append(psi_tar_n_iter)
        sigma_iter.append(sigma_n_iter)

        if n_iter == 0:
            A_ntar_n_iter = []
            B_ntar_n_iter = []
            psi_ntar_n_iter = mcmc_output_dict_n_iter['psi_ntar_0'][0, ref_mcmc_id]

            for e_it in range(E_num):
                A_ntar_n_e_iter = mcmc_output_dict_n_iter['alpha_ntar_0_{}'.format(e_it)][ref_mcmc_id, :]
                A_ntar_n_iter.append(A_ntar_n_e_iter)
                B_ntar_n_iter.append(eigen_fun_dict['non-target'] @ A_ntar_n_e_iter)

            A_ntar_n_iter = np.stack(A_ntar_n_iter, axis=0)
            B_ntar_n_iter = psi_ntar_n_iter * np.stack(B_ntar_n_iter, axis=0)

            init_param_dict['A_0_ntar'].append(A_ntar_n_iter)
            init_param_dict['psi_0_ntar'].append(psi_ntar_n_iter)
            init_param_dict['B_0_ntar'].append(B_ntar_n_iter)

    A_tar_iter = np.stack(A_tar_iter, axis=0)
    B_tar_iter = np.stack(B_tar_iter, axis=0)
    psi_tar_iter = np.stack(psi_tar_iter, axis=0)
    sigma_iter = np.stack(sigma_iter, axis=0)

    init_param_dict['A_tar'].append(A_tar_iter)
    init_param_dict['B_tar'].append(B_tar_iter)
    init_param_dict['psi_tar'].append(psi_tar_iter)
    init_param_dict['sigma'].append(sigma_iter)

    return init_param_dict


def initialize_data_multi_sigma_prior_hyper_param_eeg(
        E_total, channel_ids_str, total_sub_name, sub_new_name, seq_new_i,
        seq_source_size_2, parent_dir, xdawn_bool, letter_dim_sub, **kwargs
):

    N_total = len(total_sub_name)
    inv_gamma_a = []
    inv_gamma_scale = []

    for sub_n in total_sub_name:
        reference_n_dir = determine_proper_ref_mcmc_output_dir(
            channel_ids_str, sub_new_name, sub_n, seq_new_i, seq_source_size_2,
            parent_dir, xdawn_bool, letter_dim_sub, **kwargs
        )
        ref_n_mcmc = sio.loadmat(reference_n_dir)
        for e in range(E_total):
            sigma_n_e = ref_n_mcmc['sigma_0_{}'.format(e)]
            sigma_n_e_mean = np.mean(sigma_n_e)
            sigma_n_e_var = np.var(sigma_n_e)
            # sigma_n_e_var = 0.01
            a_n_e, scale_n_e = approx_inv_gamma_param_from_mu_and_var(sigma_n_e_mean, sigma_n_e_var)
            inv_gamma_a.append(a_n_e)
            inv_gamma_scale.append(scale_n_e)
    inv_gamma_a = np.reshape(np.stack(inv_gamma_a, axis=0), [N_total, E_total])
    inv_gamma_scale = np.reshape(np.stack(inv_gamma_scale, axis=0), [N_total, E_total])
    # print(inv_gamma_a)
    # print(inv_gamma_scale)
    return inv_gamma_a, inv_gamma_scale


def initialize_data_multi_source_sigma_param_eeg(
        E_total, channel_ids_str, total_sub_name, sub_new_name, seq_new_i,
        seq_source_size_2, parent_dir, xdawn_bool, letter_dim_sub, **kwargs
):

    N_total = len(total_sub_name)
    inv_gamma_a = []
    inv_gamma_scale = []

    for sub_n in total_sub_name:
        reference_n_dir = determine_proper_ref_mcmc_output_dir(
            channel_ids_str, sub_new_name, sub_n, seq_new_i, seq_source_size_2,
            parent_dir, xdawn_bool, letter_dim_sub, **kwargs
        )
        ref_n_mcmc = sio.loadmat(reference_n_dir)
        for e in range(E_total):
            sigma_n_e = ref_n_mcmc['sigma_0_{}'.format(e)]
            sigma_n_e_mean = np.mean(sigma_n_e)
            sigma_n_e_var = np.var(sigma_n_e)
            # sigma_n_e_var = 0.01
            a_n_e, scale_n_e = approx_inv_gamma_param_from_mu_and_var(sigma_n_e_mean, sigma_n_e_var)
            inv_gamma_a.append(a_n_e)
            inv_gamma_scale.append(scale_n_e)
    inv_gamma_a = np.reshape(np.stack(inv_gamma_a, axis=0), [N_total, E_total])
    inv_gamma_scale = np.reshape(np.stack(inv_gamma_scale, axis=0), [N_total, E_total])
    # print(inv_gamma_a)
    # print(inv_gamma_scale)
    return inv_gamma_a, inv_gamma_scale


# model fitting
def generate_mat_normal_rvs_fast(M_mat, U_mat, V_mat):
    r"""
    :param M_mat:
    :param U_mat:
    :param V_mat:
    :return:
    """
    v, p = M_mat.shape
    MN_A = np.linalg.cholesky(U_mat)
    MN_B = np.linalg.cholesky(V_mat).T
    A_mat_rvs = np.random.normal(loc=0, scale=1, size=[v, p])
    A_mat_rvs = M_mat + MN_A @ A_mat_rvs @ MN_B
    return A_mat_rvs


def create_mat_normal_obj(m_mat, row_cov_mat, col_cov_mat):
    mat_obj = matn(mean=m_mat, rowcov=row_cov_mat, colcov=col_cov_mat)
    return mat_obj


def create_mat_normal_obj_from_mcmc(m_mat, rho_val, eta_val, sigma_vec, temp_length, spatial_length):
    cov_rho_temp = create_exponential_decay_cov_mat(
        1.0, rho_val, temp_length
    )
    cov_eta_spatial = create_cs_cov_mat(
        sigma_vec ** 2, eta_val, spatial_length
    )
    mat_obj = create_mat_normal_obj(m_mat, cov_eta_spatial, cov_rho_temp)
    return mat_obj


def _compute_new_data_multi_log_likelihood(
        E, signal_length_2, input_data,
        B_tar, B_ntar, sigma, eta, rho
):
    cov_spatial = create_cs_cov_mat(sigma ** 2, eta, E)
    cov_temp = create_exponential_decay_cov_mat(1.0, rho, signal_length_2)
    matn_tar_obj = create_mat_normal_obj(B_tar, cov_spatial, cov_temp)
    matn_ntar_obj = create_mat_normal_obj(B_ntar, cov_spatial, cov_temp)

    new_data_tar_log_prob = np.sum(matn_tar_obj.logpdf(X=input_data['target']), axis=0)
    new_data_ntar_log_prob = np.sum(matn_ntar_obj.logpdf(X=input_data['non-target']), axis=0)

    return new_data_tar_log_prob, new_data_ntar_log_prob



# summary functions (after model fitting)
def signal_beta_multi_ref_summary(
        mcmc_dict: dict, eigen_fun_mat_dict: dict, suffix_name: str,
        channel_dim: int, q_low=0.05
):
    q_upp = 1 - q_low

    psi_ntar_name = 'psi_ntar_{}'.format(suffix_name)
    psi_ntar = np.squeeze(mcmc_dict[psi_ntar_name])[:, np.newaxis]

    psi_tar_name = 'psi_tar_{}'.format(suffix_name)
    psi_tar = np.squeeze(mcmc_dict[psi_tar_name])[:, np.newaxis]

    beta_tar_ls = []
    beta_ntar_ls = []
    for channel_e in range(channel_dim):
        # non-target
        alpha_ntar_e_name = 'alpha_ntar_{}_{}'.format(suffix_name, channel_e)
        alpha_ntar_e = mcmc_dict[alpha_ntar_e_name]

        beta_ntar_e = psi_ntar * np.matmul(
            alpha_ntar_e, np.transpose(eigen_fun_mat_dict['non-target'])
        )
        beta_ntar_ls.append(beta_ntar_e)

        # target
        alpha_tar_e_name = 'alpha_tar_{}_{}'.format(suffix_name, channel_e)
        alpha_tar_e = mcmc_dict[alpha_tar_e_name]

        beta_tar_e = psi_tar * np.matmul(
            alpha_tar_e, np.transpose(eigen_fun_mat_dict['target'])
        )
        beta_tar_ls.append(beta_tar_e)

    beta_ntar_ls = np.stack(beta_ntar_ls, axis=1)
    beta_ntar_mean = np.mean(beta_ntar_ls, axis=0)
    beta_ntar_low = np.quantile(beta_ntar_ls, q=q_low, axis=0)
    beta_ntar_upp = np.quantile(beta_ntar_ls, q=q_upp, axis=0)
    beta_ntar_dict = {
        'mean': beta_ntar_mean,
        'low': beta_ntar_low,
        'upp': beta_ntar_upp
    }

    beta_tar_ls = np.stack(beta_tar_ls, axis=1)
    beta_tar_mean = np.mean(beta_tar_ls, axis=0)
    beta_tar_low = np.quantile(beta_tar_ls, q=q_low, axis=0)
    beta_tar_upp = np.quantile(beta_tar_ls, q=q_upp, axis=0)

    beta_tar_dict = {
        'mean': beta_tar_mean,
        'low': beta_tar_low,
        'upp': beta_tar_upp
    }

    return beta_tar_dict, beta_ntar_dict


def signal_B_multi_borrow_summary(
        mcmc_dict: dict, suffix_name: str, q_low=0.05
):
    q_upp = 1 - q_low

    beta_ntar_ls = mcmc_dict['B_{}_ntar'.format(suffix_name)]
    beta_ntar_mean = np.mean(beta_ntar_ls, axis=0)
    beta_ntar_low = np.quantile(beta_ntar_ls, q=q_low, axis=0)
    beta_ntar_upp = np.quantile(beta_ntar_ls, q=q_upp, axis=0)
    beta_ntar_dict = {
        'mean': beta_ntar_mean,
        'low': beta_ntar_low,
        'upp': beta_ntar_upp
    }

    beta_tar_ls = mcmc_dict['B_tar'][:, 0, ...]
    # beta_tar_ls = np.stack(beta_tar, axis=1)
    beta_tar_mean = np.mean(beta_tar_ls, axis=0)
    beta_tar_low = np.quantile(beta_tar_ls, q=q_low, axis=0)
    beta_tar_upp = np.quantile(beta_tar_ls, q=q_upp, axis=0)

    beta_tar_dict = {
        'mean': beta_tar_mean,
        'low': beta_tar_low,
        'upp': beta_tar_upp
    }

    return beta_tar_dict, beta_ntar_dict


def numpyro_data_multi_reference_signal_sim_wrap_up(
        E, input_data, index_x, eigen_val_dict, eigen_fun_mat_dict, rng_key_init,
        scenario_name_dir, sub_n, seq_size, **kwargs
):
    # rng_key_init = n + 1
    signal_length = len(index_x)
    nuts_kernel = NUTS(signal_new_sim_multi)
    num_warmup = 2000
    num_samples = 200
    y_low = -4
    y_upp = 4

    if 'num_warmup' in kwargs:
        num_warmup = kwargs['num_warmup']
    if 'num_samples' in kwargs:
        num_samples = kwargs['num_samples']
    if 'y_low' in kwargs:
        y_low = kwargs['y_low']
    if 'y_upp' in kwargs:
        y_upp = kwargs['y_upp']

    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

    rng_key_iter = random.PRNGKey(rng_key_init)
    mcmc.run(
        rng_key_iter, input_points=index_x,
        eigen_val_dict=eigen_val_dict['group_0'],
        eigen_fun_mat_dict=eigen_fun_mat_dict['group_0'],
        input_data=input_data,
        E=E,
        extra_fields=('potential_energy',)
    )
    mcmc_iter_dict = mcmc.get_samples()
    mcmc.print_summary()

    # save mcmc_dict
    mcmc_iter_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(scenario_name_dir, sub_n, seq_size)
    sio.savemat(mcmc_iter_dict_dir, mcmc_iter_dict)


    # save mcmc summary using sys.stdout
    summary_dict_dir = '{}/summary_sub_{}_seq_size_{}_reference.txt'.format(scenario_name_dir, sub_n, seq_size)
    stdoutOrigin = sys.stdout
    sys.stdout = open(summary_dict_dir, "w")
    mcmc.print_summary()
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    mcmc.print_summary()
    # # load mcmc_dict
    # mcmc_summary_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(scenario_name_dir, sub_n, seq_size)
    # mcmc_iter_dict = sio.loadmat(mcmc_summary_dict_dir)

    # produce the beta function plots
    beta_tar_summary_dict, beta_ntar_summary_dict = signal_beta_multi_ref_summary(
        mcmc_iter_dict, eigen_fun_mat_dict['group_0'], str(0), E, 0.05
    )

    x_time = np.arange(signal_length)
    fig0, ax0 = plt.subplots(1, E, figsize=(12, 6))
    for e in range(E):
        ax0[e].plot(x_time, beta_ntar_summary_dict['mean'][e, :], label='non-target', color='blue')
        ax0[e].plot(x_time, beta_tar_summary_dict['mean'][e, :], label='target', color='red')
        ax0[e].fill_between(x_time, beta_ntar_summary_dict['low'][e, :],
                            beta_ntar_summary_dict['upp'][e, :], alpha=0.2, label='non-target', color='blue')
        ax0[e].fill_between(x_time, beta_tar_summary_dict['low'][e, :],
                            beta_tar_summary_dict['upp'][e, :], alpha=0.2, label='target', color='red')
        ax0[e].legend(loc='best')
        ax0[e].set_ylim([y_low, y_upp])
        ax0[e].set_xlabel('Time (unit)')
        ax0[e].set_title('Channel {}'.format(e + 1))
    fig0.savefig('{}/plot_sub_{}_seq_size_{}_reference.png'.format(scenario_name_dir, sub_n, seq_size))
    # fig0.close()

    return mcmc_iter_dict


def compute_single_seq_log_lhd_sum(pred_score, log_lhd_tar, log_lhd_ntar):
    log_lhd_sum = 0
    for i in range(rcp_unit_flash_num):
        if pred_score[i] >= 0:
            log_lhd_sum = log_lhd_sum + log_lhd_tar[i]
        else:
            log_lhd_sum = log_lhd_sum + log_lhd_ntar[i]
    return log_lhd_sum


def ml_predict_letter_log_prob_single_seq(
        char_log_prob, stimulus_score, stimulus_code, mu_tar, mu_ntar, std_common, unit_stimulus_set
):
    """
    Apply the bayesian naive dynamic stopping criterion
    :param char_log_prob:
    :param stimulus_score:
    :param stimulus_code:
    :param mu_tar:
    :param mu_ntar:
    :param std_common:
    :param unit_stimulus_set:
    :return:
    """
    char_log_prob_post = np.copy(np.log(char_log_prob))
    for s_id in range(rcp_unit_flash_num):
        for char_id in range(1, rcp_char_size + 1):
            if char_id in unit_stimulus_set[stimulus_code[s_id]-1]:
                char_log_prob_post[char_id - 1] = char_log_prob_post[char_id - 1] + \
                                                  stats.norm.logpdf(stimulus_score[s_id], loc=mu_tar, scale=std_common)
            else:
                char_log_prob_post[char_id - 1] = char_log_prob_post[char_id - 1] + \
                                                  stats.norm.logpdf(stimulus_score[s_id], loc=mu_ntar, scale=std_common)
    char_log_prob_post = char_log_prob_post - np.max(char_log_prob_post)
    char_log_prob_post = np.exp(char_log_prob_post)
    char_log_prob_post = char_log_prob_post / np.sum(char_log_prob_post)
    return char_log_prob_post


def plot_new_data_multi_xdawn(
    new_data, new_data_X_xdawn, seq_i, new_data_tar_size, sub_new_name,
    n_components, sub_new_reference_dir_2, inference_dir_2
):
    fig2, ax2 = plt.subplots(2, 4, figsize=(24, 12))
    for i in range((seq_i + 1) * 10):
        ax2[0, 0].plot(new_data['target'][i, :signal_length], color='red')
        ax2[0, 1].plot(new_data['target'][i, signal_length:], color='red')

    for j in range((seq_i + 1) * 50):
        ax2[0, 2].plot(new_data['non-target'][j, :signal_length], color='blue')
        ax2[0, 3].plot(new_data['non-target'][j, signal_length:], color='blue')

    for ax2_iter in range(4):
        ax2[0, ax2_iter].set_ylim([-4, 4])

    ax2[0, 0].set_title('{} Target, Component 1'.format((seq_i + 1) * 10))
    ax2[0, 1].set_title('{} Target, Component 2'.format((seq_i + 1) * 10))
    ax2[0, 2].set_title('{} Non-target, Component 1'.format((seq_i + 1) * 50))
    ax2[0, 3].set_title('{} Non-target, Component 2'.format((seq_i + 1) * 50))

    ax2[1, 0].plot(np.mean(new_data_X_xdawn[:new_data_tar_size, :signal_length], axis=0), color='red')
    ax2[1, 0].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, :signal_length], axis=0, q=0.05), color='red',
                   alpha=0.5)
    ax2[1, 0].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, :signal_length], axis=0, q=0.95), color='red',
                   alpha=0.5)
    ax2[1, 0].set_ylim([-2, 2])
    ax2[1, 0].set_title('Average Target, Component 1')

    ax2[1, 1].plot(np.mean(new_data_X_xdawn[:new_data_tar_size, signal_length:], axis=0), color='red')
    ax2[1, 1].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, signal_length:], axis=0, q=0.05), color='red',
                   alpha=0.5)
    ax2[1, 1].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, signal_length:], axis=0, q=0.95), color='red',
                   alpha=0.5)
    ax2[1, 1].set_ylim([-2, 2])
    ax2[1, 1].set_title('Average Target, Component 2')

    ax2[1, 2].plot(np.mean(new_data_X_xdawn[new_data_tar_size:, :signal_length], axis=0), color='blue')
    ax2[1, 2].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, :signal_length], axis=0, q=0.05), color='blue',
                   alpha=0.5)
    ax2[1, 2].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, :signal_length], axis=0, q=0.95), color='blue',
                   alpha=0.5)
    ax2[1, 2].set_ylim([-2, 2])
    ax2[1, 2].set_title('Average Non-target, Component 1')

    ax2[1, 3].plot(np.mean(new_data_X_xdawn[new_data_tar_size:, signal_length:], axis=0), color='blue')
    ax2[1, 3].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, signal_length:], axis=0, q=0.05), color='blue',
                   alpha=0.5)
    ax2[1, 3].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, signal_length:], axis=0, q=0.95), color='blue',
                   alpha=0.5)
    ax2[1, 3].set_ylim([-2, 2])
    ax2[1, 3].set_title('Average Non-target, Component 2')
    fig2.suptitle('{}, Seq {}'.format(sub_new_name, seq_i + 1), fontsize=30)
    fig2.savefig('{}/plot_seq_size_{}_first_{}_component_first_half.png'.format(
        sub_new_reference_dir_2, seq_i + 1, n_components), bbox_inches='tight', pad_inches=0.25
    )

    fig2.savefig('{}/plot_{}_seq_size_{}_first_{}_component_descriptive.png'.format(
        inference_dir_2, sub_new_name, seq_i + 1, n_components), bbox_inches='tight', pad_inches=0.25
    )


def plot_new_data_multi_xdawn_numpyro_reference_summary(
    mcmc_iter_dict, eigen_fun_mat_dict, seq_i, signal_length_2,
    xdawn_min, n_components, sub_new_reference_dir_2
):
    beta_tar_summary_dict, beta_ntar_summary_dict = signal_beta_multi_ref_summary(
        mcmc_iter_dict, eigen_fun_mat_dict['group_0'], str(0), xdawn_min, 0.05
    )
    x_time = np.arange(signal_length_2)
    beta_tar_summary_dict['mean'] = np.reshape(beta_tar_summary_dict['mean'], [xdawn_min, signal_length_2])
    beta_tar_summary_dict['upp'] = np.reshape(beta_tar_summary_dict['upp'], [xdawn_min, signal_length_2])
    beta_tar_summary_dict['low'] = np.reshape(beta_tar_summary_dict['low'], [xdawn_min, signal_length_2])

    beta_ntar_summary_dict['mean'] = np.reshape(beta_ntar_summary_dict['mean'], [xdawn_min, signal_length_2])
    beta_ntar_summary_dict['upp'] = np.reshape(beta_ntar_summary_dict['upp'], [xdawn_min, signal_length_2])
    beta_ntar_summary_dict['low'] = np.reshape(beta_ntar_summary_dict['low'], [xdawn_min, signal_length_2])

    fig0, ax0 = plt.subplots(1, xdawn_min, figsize=(15, 10))
    for filter_i in range(xdawn_min):
        ax0[filter_i].plot(x_time, beta_ntar_summary_dict['mean'][filter_i, :], label='non-target',
                           color='blue')
        ax0[filter_i].plot(x_time, beta_tar_summary_dict['mean'][filter_i, :], label='target', color='red')
        ax0[filter_i].fill_between(x_time, beta_ntar_summary_dict['low'][filter_i, :],
                                   beta_ntar_summary_dict['upp'][filter_i, :], alpha=0.2, label='non-target',
                                   color='blue')
        ax0[filter_i].fill_between(x_time, beta_tar_summary_dict['low'][filter_i, :],
                                   beta_tar_summary_dict['upp'][filter_i, :], alpha=0.2, label='target',
                                   color='red')
        ax0[filter_i].set_ylim([-3, 3])
        ax0[filter_i].legend(loc='best')
        ax0[filter_i].set_title('All Channels with {} Components'.format(n_components))
        ax0[filter_i].set_xlabel('Time (unit)')

    fig0.savefig('{}/plot_seq_size_{}_reference_xdawn.png'.format(sub_new_reference_dir_2, seq_i + 1))
    plt.close()
