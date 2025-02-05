from self_py_fun.SimFun import *
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp
import scipy.io as sio
from numpyro.infer import NUTS, MCMC


# single-channel related functions
def signal_new_sim(
        input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None
):
    r"""
    :param input_points:
    :param eigen_val_dict:
    :param eigen_fun_mat_dict:
    :param input_data:
    :return: This is a reference panel (restriction) for the new subject
    """
    signal_length = len(input_points)
    var_ntar_name = 'psi_ntar_0'
    var_ntar = numpyro.sample(var_ntar_name, dist.LogNormal(loc=0.0, scale=1.0))

    theta_ntar_name = 'alpha_ntar_0'
    theta_ntar = numpyro.sample(
        theta_ntar_name, dist.MultivariateNormal(
            loc=jnp.zeros_like(eigen_val_dict['non-target']),
            covariance_matrix=jnp.diag(eigen_val_dict['non-target'])
        )
    )

    beta_ntar = var_ntar * jnp.matmul(eigen_fun_mat_dict['non-target'], theta_ntar)

    var_tar_name = 'psi_tar_0'
    var_tar = numpyro.sample(var_tar_name, dist.LogNormal(loc=0.0, scale=2.0))

    theta_tar_name = 'alpha_tar_0'
    theta_tar = numpyro.sample(
        theta_tar_name, dist.MultivariateNormal(
            loc=jnp.zeros_like(eigen_val_dict['target']),
            covariance_matrix=jnp.diag(eigen_val_dict['target'])
        )
    )
    beta_tar = var_tar * jnp.matmul(eigen_fun_mat_dict['target'], theta_tar)

    s_name = 'sigma_0'
    sigma_val = numpyro.sample(s_name, dist.HalfCauchy(scale=5.0))
    rho_name = 'rho_0'
    rho = numpyro.sample(rho_name, dist.Beta(concentration1=1.0, concentration0=1.0))
    cov_mat = create_exponential_decay_cov_mat_jnp(sigma_val**2, rho, signal_length)

    data_0_tar_name = 'subject_0_tar'
    new_0_tar_dist = dist.MultivariateNormal(
        loc=beta_tar,
        covariance_matrix=cov_mat
    )
    numpyro.sample(data_0_tar_name, new_0_tar_dist, obs=input_data['target'])

    data_0_ntar_name = 'subject_0_ntar'
    new_0_ntar_dist = dist.MultivariateNormal(
        loc=beta_ntar,
        covariance_matrix=cov_mat
    )
    numpyro.sample(data_0_ntar_name, new_0_ntar_dist, obs=input_data['non-target'])


def import_mcmc_summary_reference(reference_mcmc_dir, eigen_fun_dict):

    reference_mcmc = sio.loadmat(reference_mcmc_dir)
    alpha_tar_0 = reference_mcmc['alpha_tar_0']
    alpha_ntar_0 = reference_mcmc['alpha_ntar_0']
    psi_tar_0 = reference_mcmc['psi_tar_0']
    psi_ntar_0 = reference_mcmc['psi_ntar_0']
    sigma_0_mean = np.mean(reference_mcmc['sigma_0'])
    rho_0_mean = np.mean(reference_mcmc['rho_0'])

    beta_tar_0 = psi_tar_0.T * alpha_tar_0 @ eigen_fun_dict['target'].T
    beta_ntar_0 = psi_ntar_0.T * alpha_ntar_0 @ eigen_fun_dict['non-target'].T
    beta_tar_0_mean = np.mean(beta_tar_0, axis=0)
    beta_ntar_0_mean = np.mean(beta_ntar_0, axis=0)

    psi_tar_0_mean = np.mean(psi_tar_0)
    psi_ntar_0_mean = np.mean(psi_ntar_0)

    # use pseudo-inverse to obtain alpha_tar_0 and alpha_ntar_0
    alpha_tar_0_mean = pinv(eigen_fun_dict['target']) @ beta_tar_0_mean / psi_tar_0_mean
    alpha_ntar_0_mean = pinv(eigen_fun_dict['non-target']) @ beta_ntar_0_mean / psi_ntar_0_mean

    return [beta_tar_0_mean, beta_ntar_0_mean, alpha_tar_0_mean, alpha_ntar_0_mean,
            psi_tar_0_mean, psi_ntar_0_mean, sigma_0_mean, rho_0_mean]


def initialize_data_fast_calculation_sim(
        N_total, signal_length_2, seq_new_i, seq_source_size_2, new_data,
        eigen_fun_mat_dict, eigen_val_dict, scenario_name_dir
):
    r"""
    :param N_total: source participant size
    :param signal_length_2:
    :param seq_new_i:
    :param seq_source_size_2:
    :param new_data:
    :param eigen_fun_mat_dict:
    :param eigen_val_dict:
    :param scenario_name_dir:
    :return:
    """
    init_param_dict = {
        "beta_tar": [np.ones([N_total, signal_length_2])],
        "beta_0_ntar": [np.ones([signal_length_2])],
        "alpha_tar": [np.ones([N_total, len(eigen_val_dict['target'])])],
        "alpha_0_ntar": [np.ones([len(eigen_val_dict['non-target'])])],
        "psi_tar": [np.ones([N_total])],
        "psi_0_ntar": [1],
        "sigma": [np.ones([N_total])],
        "rho": [np.ones([N_total]) - 0.5],
        "z_vector": [np.random.binomial(1, 0.5, size=N_total-1)],
        "z_vector_prob": [np.ones([N_total-1]) / 2],
        "log_joint_prob": [0]
    }

    beta_tar = []
    beta_0_ntar = []
    alpha_tar = []
    alpha_0_ntar = []
    psi_tar = []
    psi_0_ntar = []
    sigma = []
    rho = []
    new_tar_log_prob = []

    # may need to adapt to the directory for the simulation studies
    for n in range(N_total):
        if n == 0:
            reference_n_dir = '{}/reference_numpyro/mcmc_sub_0_seq_size_{}_reference.mat'.format(
                scenario_name_dir, seq_new_i + 1
            )
        else:
            reference_n_dir = '{}/reference_numpyro/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                scenario_name_dir, n, seq_source_size_2
            )
        [beta_tar_n, beta_ntar_n, alpha_tar_n, alpha_ntar_n,
         psi_tar_n, psi_ntar_n, sigma_n, rho_n] = import_mcmc_summary_reference(
            reference_n_dir, eigen_fun_mat_dict
        )
        _, _, new_tar_log_prob_n, new_ntar_log_prob_n = _compute_new_data_log_likelihood(signal_length_2, new_data,
                                                                                         beta_tar_n, beta_ntar_n,
                                                                                         sigma_n, rho_n)

        beta_tar.append(beta_tar_n)
        alpha_tar.append(alpha_tar_n)
        psi_tar.append(psi_tar_n)
        sigma.append(sigma_n)
        rho.append(rho_n)
        new_tar_log_prob.append(new_tar_log_prob_n)

        if n == 0:
            beta_0_ntar.append(beta_ntar_n)
            alpha_0_ntar.append(alpha_ntar_n)
            psi_0_ntar.append(psi_ntar_n)

    beta_tar = np.stack(beta_tar, axis=0)
    beta_0_ntar = np.stack(beta_0_ntar, axis=0)
    alpha_tar = np.stack(alpha_tar, axis=0)
    alpha_0_ntar = np.stack(alpha_0_ntar, axis=0)
    psi_tar = np.stack(psi_tar, axis=0)
    psi_0_ntar = np.stack(psi_0_ntar, axis=0)
    sigma = np.stack(sigma, axis=0)
    rho = np.stack(rho, axis=0)
    new_tar_log_prob = np.stack(new_tar_log_prob, axis=0)

    z_vector_init = []
    z_vector_prob_init = []
    for n in range(N_total - 1):
        z_vector_prob_n = convert_log_prob_to_prob(np.array([new_tar_log_prob[0], new_tar_log_prob[n + 1]]))[1]
        z_vector_init.append(np.random.binomial(1, z_vector_prob_n, size=None))
        z_vector_prob_init.append(z_vector_prob_n)

    z_vector_init = np.stack(z_vector_init, axis=0)
    z_vector_prob_init = np.stack(z_vector_prob_init, axis=0)
   
    init_param_dict['z_vector'][0] = np.copy(z_vector_init)
    init_param_dict['z_vector_prob'][0] = np.copy(z_vector_prob_init)

    init_param_dict['beta_tar'][0] = np.copy(beta_tar)
    init_param_dict['alpha_tar'][0] = np.copy(alpha_tar)
    init_param_dict['psi_tar'][0] = np.copy(psi_tar)

    init_param_dict['beta_0_ntar'][0] = np.squeeze(beta_0_ntar, axis=0)
    init_param_dict['alpha_0_ntar'][0] = np.squeeze(alpha_0_ntar, axis=0)
    init_param_dict['psi_0_ntar'][0] = np.squeeze(psi_0_ntar)

    init_param_dict['sigma'][0] = np.copy(sigma)
    init_param_dict['rho'][0] = np.copy(rho)

    # borrow_dict_dir = '{}/borrow_gibbs/mcmc_sub_0_seq_size_{}_cluster.mat'.format(
    #     scenario_name_dir, seq_new_i + 1
    # )
    # borrow_mcmc_dict = sio.loadmat(borrow_dict_dir)
    #
    # init_param_dict['z_vector'][0] = np.copy(borrow_mcmc_dict['z_vector'][0, :])
    # init_param_dict['z_vector_prob'][0] = np.copy(borrow_mcmc_dict['z_vector_prob'][0, :])
    #
    # init_param_dict['beta_tar'][0] = np.copy(borrow_mcmc_dict['beta_tar'][0, ...])
    # init_param_dict['alpha_tar'][0] = np.copy(borrow_mcmc_dict['alpha_tar'][0, ...])
    # init_param_dict['psi_tar'][0] = np.copy(borrow_mcmc_dict['psi_tar'][0, :])
    #
    # init_param_dict['beta_0_ntar'][0] = np.copy(borrow_mcmc_dict['beta_0_ntar'][0, :])
    # init_param_dict['alpha_0_ntar'][0] = np.copy(borrow_mcmc_dict['alpha_0_ntar'][0, :])
    # init_param_dict['psi_0_ntar'][0] = borrow_mcmc_dict['psi_0_ntar'][0, 0]
    #
    # init_param_dict['sigma'][0] = np.copy(borrow_mcmc_dict['sigma'][0, :])
    # init_param_dict['rho'][0] = np.copy(borrow_mcmc_dict['rho'][0, :])

    return init_param_dict


def initialize_data_fast_calculation_eeg_xdawn(
        N_total, signal_length_2,
        seq_new_i, sub_new_name, new_data,
        seq_source_size_2, letter_dim_sub, n_components,
        total_sub_name, eigen_fun_mat_dict, eigen_val_dict, parent_dir
):

    init_param_dict = {
        "beta_tar": [np.ones([N_total, signal_length_2])],
        "beta_0_ntar": [np.ones([signal_length_2])],
        "alpha_tar": [np.ones([N_total, len(eigen_val_dict['target'])])],
        "alpha_0_ntar": [np.ones([len(eigen_val_dict['non-target'])])],
        "psi_tar": [np.ones([N_total])],
        "psi_0_ntar": [1],
        "sigma": [np.ones([N_total])],
        "rho": [np.ones([N_total]) - 0.5],
        "z_vector": [np.random.binomial(1, 0.5, size=N_total-1)],
        "z_vector_prob": [np.ones([N_total-1]) / 2],
        "log_joint_prob": [0]
    }

    beta_tar_N = []
    beta_ntar_N = []
    alpha_tar_N = []
    alpha_ntar_N = []
    psi_tar_N = []
    psi_ntar_N = []
    sigma_N = []
    rho_N = []
    new_log_prob_tar_N = []

    # may need to adapt to the directory for the simulation studies
    for sub_n in total_sub_name:
        if sub_n == sub_new_name:
            reference_n_dir = '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                parent_dir, sub_n, letter_dim_sub, n_components, seq_new_i + 1
            )
        else:
            reference_n_dir = '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                parent_dir, sub_n, letter_dim_sub, n_components, seq_source_size_2
            )

        [beta_tar_n, beta_ntar_n, alpha_tar_n, alpha_ntar_n,
         psi_tar_n, psi_ntar_n, sigma_n, rho_n] = import_mcmc_summary_reference(
            reference_n_dir, eigen_fun_mat_dict
        )
        _, _, new_tar_log_prob_n, new_ntar_log_prob_n = _compute_new_data_log_likelihood(signal_length_2, new_data,
                                                                                         beta_tar_n, beta_ntar_n,
                                                                                         sigma_n, rho_n)

        beta_tar_N.append(beta_tar_n)
        alpha_tar_N.append(alpha_tar_n)
        psi_tar_N.append(psi_tar_n)
        sigma_N.append(sigma_n)
        rho_N.append(rho_n)
        new_log_prob_tar_N.append(new_tar_log_prob_n)

        if sub_n == sub_new_name:
            alpha_ntar_N.append(alpha_ntar_n)
            beta_ntar_N.append(beta_ntar_n)
            psi_ntar_N.append(psi_ntar_n)

    beta_tar_N = np.stack(beta_tar_N, axis=0)
    beta_ntar_N = np.stack(beta_ntar_N, axis=0)
    alpha_tar_N = np.stack(alpha_tar_N, axis=0)
    alpha_ntar_N = np.stack(alpha_ntar_N, axis=0)
    psi_tar_N = np.stack(psi_tar_N, axis=0)
    psi_ntar_N = np.stack(psi_ntar_N, axis=0)
    sigma_N = np.stack(sigma_N, axis=0)
    rho_N = np.stack(rho_N, axis=0)
    new_log_prob_tar_N = np.stack(new_log_prob_tar_N, axis=0)

    z_vector_init = []
    z_vector_prob_init = []
    for n in range(N_total - 1):
        z_vector_prob_n = convert_log_prob_to_prob(np.array([new_log_prob_tar_N[0], new_log_prob_tar_N[n + 1]]))[1]
        z_vector_init.append(np.random.binomial(1, z_vector_prob_n, size=None))
        z_vector_prob_init.append(z_vector_prob_n)
    z_vector_init = np.stack(z_vector_init, axis=0)
    z_vector_prob_init = np.stack(z_vector_prob_init, axis=0)

    init_param_dict['z_vector'][0] = np.copy(z_vector_init)
    init_param_dict['z_vector_prob'][0] = np.copy(z_vector_prob_init)

    init_param_dict['beta_tar'][0] = np.copy(beta_tar_N)
    init_param_dict['alpha_tar'][0] = np.copy(alpha_tar_N)
    init_param_dict['psi_tar'][0] = np.copy(psi_tar_N)

    init_param_dict['beta_0_ntar'][0] = np.squeeze(beta_ntar_N, axis=0)
    init_param_dict['alpha_0_ntar'][0] = np.squeeze(alpha_ntar_N, axis=0)
    init_param_dict['psi_0_ntar'][0] = np.copy(psi_ntar_N)

    init_param_dict['sigma'][0] = np.copy(sigma_N)
    init_param_dict['rho'][0] = np.copy(rho_N)

    return init_param_dict


def _compute_new_data_log_likelihood(
        signal_length_2, new_data,
        beta_tar, beta_ntar, sigma, rho,
):
    cov_mat = create_exponential_decay_cov_mat(sigma ** 2, rho, signal_length_2)
    matn_tar_obj = mvn(mean=beta_tar, cov=cov_mat)
    matn_ntar_obj = mvn(mean=beta_ntar, cov=cov_mat)
    new_data_tar_log_prob = np.sum(matn_tar_obj.logpdf(x=new_data['target']), axis=0)
    new_data_ntar_log_prob = np.sum(matn_ntar_obj.logpdf(x=new_data['non-target']), axis=0)

    return matn_tar_obj, matn_ntar_obj, new_data_tar_log_prob, new_data_ntar_log_prob


def create_mvn_object(alpha_vec, phi_mat, psi_var, sigma_val, rho_val, signal_length):

    beta_vec = psi_var * np.matmul(phi_mat, alpha_vec)
    cov_mat = create_exponential_decay_cov_mat(sigma_val ** 2, rho_val, signal_length)
    mvn_obj = mvn(mean=beta_vec, cov=cov_mat)
    return mvn_obj


def create_mvn_object_easy(beta_vec, sigma_val, rho_val, signal_length):

    # beta_vec = psi_var * np.matmul(phi_mat, alpha_vec)
    cov_mat = create_exponential_decay_cov_mat(sigma_val ** 2, rho_val, signal_length)
    mvn_obj = mvn(mean=beta_vec, cov=cov_mat)
    return mvn_obj


def signal_beta_ref_summary(
        mcmc_dict: dict, eigen_fun_mat_dict: dict, suffix_name: str, q_low=0.05
):
    q_upp = 1 - q_low

    alpha_ntar_name = 'alpha_ntar_{}'.format(suffix_name)
    alpha_ntar = mcmc_dict[alpha_ntar_name]
    psi_ntar_name = 'psi_ntar_{}'.format(suffix_name)
    psi_ntar = mcmc_dict[psi_ntar_name]

    if len(psi_ntar.shape) == 2:
        psi_ntar = np.transpose(psi_ntar)
    else:
        psi_ntar = psi_ntar[:, np.newaxis]

    beta_ntar = psi_ntar * np.matmul(
        alpha_ntar, np.transpose(eigen_fun_mat_dict['non-target'])
    )

    beta_ntar_mean = np.mean(beta_ntar, axis=0)
    beta_ntar_low = np.quantile(beta_ntar, q=q_low, axis=0)
    beta_ntar_upp = np.quantile(beta_ntar, q=q_upp, axis=0)

    beta_ntar_dict = {
        'mean': beta_ntar_mean,
        'low': beta_ntar_low,
        'upp': beta_ntar_upp
    }

    alpha_tar_name = 'alpha_tar_{}'.format(suffix_name)
    alpha_tar = mcmc_dict[alpha_tar_name]
    psi_tar_name = 'psi_tar_{}'.format(suffix_name)
    psi_tar = mcmc_dict[psi_tar_name]

    if len(psi_tar.shape) == 2:
        psi_tar = np.transpose(psi_tar)
    else:
        psi_tar = psi_tar[:, np.newaxis]

    beta_tar = psi_tar * np.matmul(
        alpha_tar, np.transpose(eigen_fun_mat_dict['target'])
    )
    beta_tar_mean = np.mean(beta_tar, axis=0)
    beta_tar_low = np.quantile(beta_tar, q=q_low, axis=0)
    beta_tar_upp = np.quantile(beta_tar, q=q_upp, axis=0)

    beta_tar_dict = {
        'mean': beta_tar_mean,
        'low': beta_tar_low,
        'upp': beta_tar_upp
    }

    return beta_tar_dict, beta_ntar_dict


def numpyro_data_reference_signal_sim_wrap_up(
        input_data, index_x, eigen_val_dict, eigen_fun_mat_dict, rng_key_init,
        scenario_name_dir, sub_n, seq_size,
        num_warmup=2000, num_samples=200
):
    # rng_key_init = n + 1
    signal_length = len(index_x)
    nuts_kernel = NUTS(signal_new_sim)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

    rng_key_iter = random.PRNGKey(rng_key_init)
    mcmc.run(
        rng_key_iter, input_points=index_x,
        eigen_val_dict=eigen_val_dict['group_0'],
        eigen_fun_mat_dict=eigen_fun_mat_dict['group_0'],
        input_data=input_data,
        extra_fields=('potential_energy',)
    )
    mcmc_iter_dict = mcmc.get_samples()
    mcmc.print_summary()

    # save mcmc_dict
    mcmc_iter_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(scenario_name_dir, sub_n, seq_size)
    # on the server, we need to manually convert it to DeviceArray.
    for mcmc_iter_dict_key in mcmc_iter_dict.keys():
        mcmc_iter_dict[mcmc_iter_dict_key] = np.asarray(mcmc_iter_dict[mcmc_iter_dict_key])
    sio.savemat(mcmc_iter_dict_dir, mcmc_iter_dict)

    '''
    # save mcmc summary using sys.stdout
    summary_dict_dir = '{}/summary_sub_{}_seq_size_{}_reference.txt'.format(scenario_name_dir, sub_n, seq_size)
    stdoutOrigin = sys.stdout
    sys.stdout = open(summary_dict_dir, "w")
    mcmc.print_summary()
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    mcmc.print_summary()

    # load mcmc_dict
    mcmc_summary_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(scenario_name_dir, sub_n, seq_size)
    mcmc_iter_dict = sio.loadmat(mcmc_summary_dict_dir)
    '''

    # produce the beta function plots
    beta_tar_summary_dict, beta_ntar_summary_dict = signal_beta_ref_summary(
        mcmc_iter_dict, eigen_fun_mat_dict['group_0'], str(0)
    )

    plt.figure()
    x_time = np.arange(signal_length)
    plt.plot(x_time, beta_ntar_summary_dict['mean'], label='non-target', color='blue')
    plt.plot(x_time, beta_tar_summary_dict['mean'], label='target', color='red')
    plt.fill_between(x_time, beta_ntar_summary_dict['low'],
                     beta_ntar_summary_dict['upp'], alpha=0.2, label='non-target', color='blue')
    plt.fill_between(x_time, beta_tar_summary_dict['low'],
                     beta_tar_summary_dict['upp'], alpha=0.2, label='target', color='red')
    plt.legend(loc='best')
    plt.xlabel('Time (unit)')
    plt.ylim([-4, 4])
    plt.savefig('{}/plot_sub_{}_seq_size_{}_reference.png'.format(scenario_name_dir, sub_n, seq_size))
    plt.close()

    return mcmc_iter_dict


def predict_reference_fast(
        mcmc_output_dict, signal_length_2, eigen_fun_mat,
        new_subject_signal
):
    # only consider posterior mean to make a fast approximation
    alpha_tar = mcmc_output_dict['alpha_tar_0']
    alpha_ntar = mcmc_output_dict['alpha_ntar_0']
    psi_tar_0 = mcmc_output_dict['psi_tar_0']
    psi_ntar_0 = mcmc_output_dict['psi_ntar_0']

    beta_tar_0 = psi_tar_0.T * alpha_tar @ eigen_fun_mat['target'].T
    beta_ntar_0 = psi_ntar_0.T * alpha_ntar @ eigen_fun_mat['non-target'].T

    rho_0 = mcmc_output_dict['rho_0']
    sigma_0 = mcmc_output_dict['sigma_0']

    beta_tar_0_mean = np.mean(beta_tar_0, axis=0)
    beta_ntar_0_mean = np.mean(beta_ntar_0, axis=0)
    rho_0_mean = np.mean(rho_0)
    sigma_0_mean = np.mean(sigma_0)

    # plt.plot(np.arange(signal_length_2), beta_tar_0_mean, label='Target', color='red')
    # plt.plot(np.arange(signal_length_2), beta_ntar_0_mean, label='Non-target', color='blue')
    # plt.legend(loc='best')
    # plt.show()

    cov_mat_0_est = create_exponential_decay_cov_mat(
        sigma_0_mean**2, rho_0_mean, signal_length_2
    )
    log_lhd_ntar_0 = mvn.logpdf(
        x=new_subject_signal, mean=beta_ntar_0_mean, cov=cov_mat_0_est
    )
    log_lhd_tar_0 = mvn.logpdf(
        x=new_subject_signal, mean=beta_tar_0_mean, cov=cov_mat_0_est
    )
    return log_lhd_tar_0 - log_lhd_ntar_0


def plot_new_data_xdawn(
        new_data, new_data_X_xdawn, seq_i, new_data_tar_size, sub_new_name,
        n_components, sub_new_reference_dir_2, inference_dir_2
):
    fig2, ax2 = plt.subplots(2, 2, figsize=(24, 12))
    for i in range((seq_i + 1) * 10):
        ax2[0, 0].plot(new_data['target'][i, :], color='red')

    for j in range((seq_i + 1) * 50):
        ax2[0, 1].plot(new_data['non-target'][j, :], color='blue')

    ax2[0, 0].set_ylim([-4, 4])
    ax2[0, 1].set_ylim([-4, 4])

    ax2[0, 0].set_title('{} Target, Component 1'.format((seq_i + 1) * 10))
    ax2[0, 1].set_title('{} Non-target, Component 1'.format((seq_i + 1) * 50))

    ax2[1, 0].plot(np.mean(new_data_X_xdawn[:new_data_tar_size, :], axis=0), color='red')
    ax2[1, 0].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, :], axis=0, q=0.05), color='red',
                   alpha=0.5)
    ax2[1, 0].plot(np.quantile(new_data_X_xdawn[:new_data_tar_size, :], axis=0, q=0.95), color='red',
                   alpha=0.5)
    ax2[1, 0].set_ylim([-2, 2])
    ax2[1, 0].set_title('Average Target, Component 1')
    ax2[1, 1].plot(np.mean(new_data_X_xdawn[new_data_tar_size:, :], axis=0), color='blue')
    ax2[1, 1].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, :], axis=0, q=0.05), color='blue',
                   alpha=0.5)
    ax2[1, 1].plot(np.quantile(new_data_X_xdawn[new_data_tar_size:, :], axis=0, q=0.95), color='blue',
                   alpha=0.5)
    ax2[1, 1].set_ylim([-2, 2])
    ax2[1, 1].set_title('Average Non-target, Component 1')

    fig2.suptitle('{}, Seq {}'.format(sub_new_name, seq_i + 1), fontsize=30)
    fig2.savefig('{}/plot_seq_size_{}_first_{}_component_first_half.png'.format(
        sub_new_reference_dir_2, seq_i + 1, n_components), bbox_inches='tight', pad_inches=0.25
    )

    fig2.savefig('{}/plot_{}_seq_size_{}_first_{}_component_descriptive.png'.format(
        inference_dir_2, sub_new_name, seq_i + 1, n_components), bbox_inches='tight', pad_inches=0.25
    )


def plot_new_data_xdawn_numpyro_reference_summary(
    mcmc_iter_dict, eigen_fun_mat_dict, seq_i, signal_length_2, sub_new_reference_dir_2
):
    # produce the beta function plots
    beta_tar_summary_dict, beta_ntar_summary_dict = signal_beta_ref_summary(
        mcmc_iter_dict, eigen_fun_mat_dict['group_0'], str(0)
    )
    x_time = np.arange(signal_length_2)
    plt.figure()
    # x_time = np.arange(signal_length)
    plt.plot(x_time, beta_ntar_summary_dict['mean'], label='non-target', color='blue')
    plt.plot(x_time, beta_tar_summary_dict['mean'], label='target', color='red')
    plt.fill_between(x_time, beta_ntar_summary_dict['low'],
                     beta_ntar_summary_dict['upp'], alpha=0.2, label='non-target', color='blue')
    plt.fill_between(x_time, beta_tar_summary_dict['low'],
                     beta_tar_summary_dict['upp'], alpha=0.2, label='target', color='red')
    plt.legend(loc='best')
    plt.xlabel('Time (unit)')
    plt.ylim([-3, 3])
    plt.savefig('{}/plot_seq_size_{}_reference_xdawn.png'.format(sub_new_reference_dir_2, seq_i + 1))
    plt.close()