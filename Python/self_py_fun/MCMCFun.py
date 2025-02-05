from self_py_fun.MCMCRefFun import *
from self_py_fun.Misc import *
import numpyro
import numpyro.distributions as dist
from scipy.stats import dirichlet
from scipy.stats import multinomial as mtn
from scipy.stats import multivariate_normal as mvn
from scipy.stats import lognorm
from scipy.stats import halfcauchy
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import norm


# single-channel related functions
def signal_integration_sim(
        N, K, input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None,
        pi_random=None
):
    signal_length_2 = len(input_points)
    psi_ntar_dict = {}
    psi_tar_dict = {}
    alpha_ntar_dict = {}
    alpha_tar_dict = {}

    beta_ntar_dict = {}
    beta_tar_dict = {}
    sigma_dict = {}
    rho_dict = {}
    cov_mat_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)

        psi_ntar_k_name = 'psi_ntar_{}'.format(k)
        psi_ntar_k = numpyro.sample(psi_ntar_k_name, dist.LogNormal(loc=0.0, scale=1.0))
        # psi_ntar_k = numpyro.sample(psi_ntar_k_name, dist.TruncatedNormal(loc=0.0, scale=5.0, low=0.0, high=10.0))
        psi_ntar_dict[psi_ntar_k_name] = psi_ntar_k

        alpha_ntar_k_name = 'alpha_ntar_{}'.format(k)
        alpha_ntar_k = numpyro.sample(
            alpha_ntar_k_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict[group_k_name]['non-target']),
                covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['non-target'])
            )
        )
        alpha_ntar_dict[alpha_ntar_k_name] = alpha_ntar_k
        beta_ntar_k = psi_ntar_k * jnp.matmul(eigen_fun_mat_dict[group_k_name]['non-target'], alpha_ntar_k)
        beta_ntar_k_name = 'beta_ntar_{}'.format(k)
        beta_ntar_dict[beta_ntar_k_name] = beta_ntar_k

        psi_tar_k_name = 'psi_tar_{}'.format(k)
        psi_tar_k = numpyro.sample(psi_tar_k_name, dist.LogNormal(loc=0.0, scale=1.0))
        # psi_tar_k = numpyro.sample(psi_tar_k_name, dist.TruncatedNormal(loc=0.0, scale=5.0, low=0.0, high=10.0))
        psi_tar_dict[psi_tar_k_name] = psi_tar_k

        alpha_tar_k_name = 'alpha_tar_{}'.format(k)
        alpha_tar_k = numpyro.sample(
            alpha_tar_k_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict[group_k_name]['target']),
                covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['target'])
            )
        )
        alpha_tar_dict[alpha_tar_k_name] = alpha_tar_k
        beta_tar_k = psi_tar_k * jnp.matmul(eigen_fun_mat_dict[group_k_name]['target'], alpha_tar_k)
        beta_tar_k_name = 'beta_tar_{}'.format(k)
        beta_tar_dict[beta_tar_k_name] = beta_tar_k

        sigma_k_name = 'sigma_{}'.format(k)
        sigma_k = numpyro.sample(sigma_k_name, dist.HalfCauchy(scale=5.0))
        sigma_dict[sigma_k_name] = sigma_k

        rho_k_name = 'rho_{}'.format(k)
        rho_k = numpyro.sample(rho_k_name, dist.Beta(concentration0=1.0, concentration1=1.0))
        rho_dict[rho_k_name] = rho_k
        cov_mat_k = create_exponential_decay_cov_mat_jnp(sigma_k ** 2, rho_k, signal_length_2)
        cov_mat_k_name = 'cov_mat_{}'.format(k)
        cov_mat_dict[cov_mat_k_name] = cov_mat_k

    # new subject component, non-target:
    data_0_ntar_name = 'subject_0_ntar'
    new_0_ntar_dist = dist.MultivariateNormal(
        loc=beta_ntar_dict['beta_ntar_0'],
        covariance_matrix=cov_mat_dict['cov_mat_0']
    )
    numpyro.sample(data_0_ntar_name, new_0_ntar_dist, obs=input_data['subject_0']['non-target'])

    # new subject component, target:
    data_0_tar_name = 'subject_0_tar'
    new_0_tar_dist = dist.MultivariateNormal(
        loc=beta_tar_dict['beta_tar_0'],
        covariance_matrix=cov_mat_dict['cov_mat_0']
    )
    numpyro.sample(data_0_tar_name, new_0_tar_dist, obs=input_data['subject_0']['target'])

    # initialize the component distribution of the mixture model by labels
    beta_tar_mix = []
    beta_ntar_mix = []
    cov_mat_mix = []
    for k in range(K):
        beta_tar_mix.append(beta_tar_dict['beta_tar_{}'.format(k)])
        beta_ntar_mix.append(beta_ntar_dict['beta_ntar_{}'.format(k)])
        cov_mat_mix.append(cov_mat_dict['cov_mat_{}'.format(k)])

    beta_tar_mix = jnp.stack(beta_tar_mix, axis=0)
    beta_ntar_mix = jnp.stack(beta_ntar_mix, axis=0)
    cov_mat_mix = jnp.stack(cov_mat_mix, axis=0)

    component_ntar_dist = dist.MultivariateNormal(
        loc=beta_ntar_mix, covariance_matrix=cov_mat_mix
    )
    component_tar_dist = dist.MultivariateNormal(
        loc=beta_tar_mix, covariance_matrix=cov_mat_mix
    )

    if pi_random:
        for n in range(N - 1):
            prob_n_name = 'prob_{}'.format(n + 1)
            prob_n = numpyro.sample(prob_n_name, dist.Dirichlet(concentration=jnp.ones(K)))
            mixing_n = dist.Categorical(probs=prob_n)

            data_n_name = 'subject_{}'.format(n + 1)
            mixture_n_ntar_dist = dist.MixtureSameFamily(mixing_n, component_ntar_dist)
            numpyro.sample('{}_ntar'.format(data_n_name), mixture_n_ntar_dist,
                           obs=input_data[data_n_name]['non-target'])

            mixture_n_tar_dist = dist.MixtureSameFamily(mixing_n, component_tar_dist)
            numpyro.sample('{}_tar'.format(data_n_name), mixture_n_tar_dist, obs=input_data[data_n_name]['target'])
    else:
        pi_prob = numpyro.sample("pi_prob", dist.Dirichlet(concentration=jnp.ones(K) * 10))
        mixing_n = dist.Categorical(probs=pi_prob)

        for n in range(N - 1):
            data_n_name = 'subject_{}'.format(n + 1)
            mixture_n_ntar_dist = dist.MixtureSameFamily(mixing_n, component_ntar_dist)
            numpyro.sample('{}_ntar'.format(data_n_name), mixture_n_ntar_dist,
                           obs=input_data[data_n_name]['non-target'])

            mixture_n_tar_dist = dist.MixtureSameFamily(mixing_n, component_tar_dist)
            numpyro.sample('{}_tar'.format(data_n_name), mixture_n_tar_dist, obs=input_data[data_n_name]['target'])


def signal_beta_integration_summary(
        mcmc_dict, K, eigen_fun_mat_dict
):
    beta_ntar_summary_dict = {}
    beta_tar_summary_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        beta_tar_k_dict, beta_ntar_k_dict = signal_beta_ref_summary(
            mcmc_dict, eigen_fun_mat_dict[group_k_name], str(k)
        )
        beta_tar_summary_dict[group_k_name] = beta_tar_k_dict
        beta_ntar_summary_dict[group_k_name] = beta_ntar_k_dict

    return beta_tar_summary_dict, beta_ntar_summary_dict


def predict_cluster_fast(
        mcmc_output_dict, signal_length_2,
        new_subject_signal, mcmc_ids
):
    # only consider posterior mean to make a fast approximation
    # remove outliers based on sigma values for group 1 to group K-1.
    if mcmc_ids is None:
        beta_tar = mcmc_output_dict['beta_tar']
        beta_0_ntar = mcmc_output_dict['beta_0_ntar']
        rho_0 = mcmc_output_dict['rho']
        sigma = mcmc_output_dict['sigma']
    else:
        beta_tar = mcmc_output_dict['beta_tar'][mcmc_ids == 1, ...]
        beta_0_ntar = mcmc_output_dict['beta_0_ntar'][mcmc_ids == 1, ...]
        rho_0 = mcmc_output_dict['rho'][mcmc_ids == 1, :]
        sigma = mcmc_output_dict['sigma'][mcmc_ids == 1, :]

    beta_tar_0_mean = np.mean(beta_tar, axis=0)[0, :]
    beta_ntar_0_mean = np.mean(beta_0_ntar, axis=0)
    rho_0_mean = np.mean(rho_0, axis=0)[0]
    sigma_0_mean = np.mean(sigma, axis=0)[0]

    # plt.plot(np.arange(signal_length_2), beta_tar_0_mean, label='Target', color='red')
    # plt.plot(np.arange(signal_length_2), beta_ntar_0_mean, label='Non-target', color='blue')
    # plt.legend(loc='best')
    # plt.show()

    cov_mat_0_est = create_exponential_decay_cov_mat(
        sigma_0_mean ** 2, rho_0_mean, signal_length_2
    )
    log_lhd_ntar_0 = mvn.logpdf(
        x=new_subject_signal, mean=beta_ntar_0_mean, cov=cov_mat_0_est
    )
    log_lhd_tar_0 = mvn.logpdf(
        x=new_subject_signal, mean=beta_tar_0_mean, cov=cov_mat_0_est
    )
    # score_0 = 1 / (1 + np.exp(log_lhd_ntar_0 - log_lhd_tar_0))
    return log_lhd_tar_0 - log_lhd_ntar_0


# The following functions apply to the Gibbs Sampler method.
def update_gibbs_sampler_per_iteration(
        eigen_fun_mat_dict, eigen_val_dict,
        source_data, new_data, N_total, signal_length_2, source_name_ls,
        # hyper-parameter updates,
        # psi_loc, psi_scale, sigma_loc, sigma_scale, rho_tf_loc, rho_tf_scale,
        step_size_ls, estimate_rho_bool, seq_source_size_2, scenario_dir,
        # parameter updates
        *args, xdawn_bool=False, **kwargs
):
    r"""
    :param xdawn_bool: bool
    :param eigen_fun_mat_dict: K-dict, each element is a 2-dict: 2d-array, (signal_length_2, d), 0 < d <= signal_length_2
    :param eigen_val_dict: K-dict, each element is a 2-dict: 1d-array, (d,)
    :param source_data: dict of source data, stratified by subject name, and target/non-target
    :param new_data: dict of new data, stratified by target/non-target
    :param N_total: integer, source + new
    :param signal_length_2: integer, feature vector length
    :param source_name_ls: list of source subjects' names
    :param step_size_ls: list psi_tar_iter, sigma_iter, and psi_0_ntar_iter
    :param estimate_rho_bool: bool variable
    :param seq_source_size_2: integer.
    :param scenario_dir: dir
    :param args: parameter updates
    :return: a new set of samples of the parameters
    """
    # update cluster-specific parameters
    psi_tar_accept = []
    psi_0_ntar_accept = []
    sigma_accept = []
    # rho_accept = []
    accept_iter_ls = []

    [beta_tar_iter, beta_0_ntar_iter, alpha_tar_iter, alpha_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter,
     sigma_iter, rho_iter, z_vector_iter, z_vector_prob_iter] = args

    beta_tar_iter = np.zeros_like(beta_tar_iter)
    beta_0_ntar_iter = np.zeros_like(beta_0_ntar_iter)

    # this may be generalized to group-specific kernel hyper-parameters
    eigen_fun_mat_tar = eigen_fun_mat_dict['target']
    eigen_val_tar = eigen_val_dict['target']
    eigen_fun_mat_ntar = eigen_fun_mat_dict['non-target']
    eigen_val_ntar = eigen_val_dict['non-target']
    ref_mcmc_id = np.random.random_integers(low=1, high=200, size=None) - 1

    # sigma_iter = np.array([3,4,4,4,3,3,3.0])

    for n in range(N_total):
        # update target-related parameters, order does not matter in Gibbs sampling.
        alpha_tar_iter = update_alpha_n_tar(alpha_tar_iter, psi_tar_iter, sigma_iter, rho_iter, z_vector_iter,
                                            eigen_fun_mat_tar, eigen_val_tar, source_data, new_data, N_total, n,
                                            signal_length_2, seq_source_size_2, scenario_dir, ref_mcmc_id,
                                            source_name_ls, xdawn_bool=xdawn_bool, **kwargs)

        # update psi_n_tar
        # psi_tar_iter = np.ones(N + 1); psi_n_tar_accept = 0.0
        psi_tar_iter, psi_n_tar_accept = update_psi_n_tar_RWMH(alpha_tar_iter, psi_tar_iter, sigma_iter, rho_iter,
                                                               z_vector_iter, eigen_fun_mat_tar, source_data, new_data,
                                                               N_total, n, signal_length_2, seq_source_size_2,
                                                               scenario_dir, ref_mcmc_id, step_size_ls[0][n],
                                                               source_name_ls, xdawn_bool=xdawn_bool, **kwargs)

        beta_tar_iter[n, :] = psi_tar_iter[n] * np.matmul(eigen_fun_mat_tar, alpha_tar_iter[n, :])

        if n == 0:
            # alpha_0_ntar_iter = np.copy(alpha_0_ntar_iter)
            alpha_0_ntar_iter = update_alpha_0_ntar(psi_0_ntar_iter, sigma_iter, rho_iter, eigen_fun_mat_ntar,
                                                    eigen_val_ntar, new_data, signal_length_2)

            # update psi_0_ntar
            # psi_0_ntar_iter = np.copy(psi_0_ntar_iter); psi_0_ntar_accept = 0.0
            psi_0_ntar_iter, psi_0_ntar_accept = update_psi_0_ntar_RWMH(alpha_0_ntar_iter, psi_0_ntar_iter,
                                                                        sigma_iter, rho_iter,
                                                                        eigen_fun_mat_ntar, new_data,
                                                                        signal_length_2, step_size_ls[2],
                                                                        **kwargs)

            beta_0_ntar_iter = psi_0_ntar_iter * np.matmul(eigen_fun_mat_ntar, alpha_0_ntar_iter)

        # update sigma_n_iter
        # sigma_n_iter = sigma_iter[n]; sigma_n_accept = 0.0
        sigma_iter, sigma_n_accept = update_sigma_n_RWMH(beta_tar_iter, beta_0_ntar_iter, sigma_iter, rho_iter,
                                                         z_vector_iter, source_data, new_data, N_total, n,
                                                         signal_length_2, seq_source_size_2, scenario_dir, ref_mcmc_id,
                                                         step_size_ls[1][n], source_name_ls, xdawn_bool=xdawn_bool,
                                                         **kwargs)

        # update rho_k_iter by discrete uniform
        if estimate_rho_bool:
            rho_iter = update_rho_n_discrete_uniform(beta_tar_iter, beta_0_ntar_iter, sigma_iter, rho_iter,
                                                     z_vector_iter, source_data, new_data, N_total, n, signal_length_2,
                                                     seq_source_size_2, scenario_dir, ref_mcmc_id, source_name_ls,
                                                     xdawn_bool=xdawn_bool, **kwargs)

        psi_tar_accept.append(psi_n_tar_accept)
        sigma_accept.append(sigma_n_accept)
        # rho_accept.append(rho_n_accept)

    # print('rho_iter={}'.format(rho_iter))
    psi_tar_accept = np.stack(psi_tar_accept, axis=0)
    sigma_accept = np.stack(sigma_accept, axis=0)
    # rho_accept = np.stack(rho_accept, axis=0)
    psi_0_ntar_accept = np.array(psi_0_ntar_accept)

    # update indicator Z
    z_vector_iter_2 = np.zeros_like(z_vector_iter)
    z_vector_prob_iter_2 = np.zeros_like(z_vector_prob_iter)

    beta_0_tar_iter = np.copy(beta_tar_iter[0, :])
    sigma_0_iter = np.copy(sigma_iter[0])
    rho_0_iter = np.copy(rho_iter[0])

    for nn in range(N_total-1):
        if source_name_ls is None:
            name_n = 'subject_{}'.format(nn + 1)
        else:
            name_n = source_name_ls[nn + 1]
        source_data_n = source_data[name_n]
        # import mcmc from reference methods:
        # mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
        #     scenario_dir, n+1, seq_source_size_2)
        # )
        #
        # alpha_n_tar_ref_iter = np.copy(mcmc_output_dict['alpha_tar_0'][ref_mcmc_id, :])
        # psi_n_tar_ref_iter = np.copy(np.squeeze(mcmc_output_dict['psi_tar_0'])[ref_mcmc_id])
        # beta_n_tar_ref_iter = psi_n_tar_ref_iter * eigen_fun_mat_tar @ alpha_n_tar_ref_iter
        # sigma_n_ref_iter = np.copy(np.squeeze(mcmc_output_dict['sigma_0'])[ref_mcmc_id])
        # rho_n_ref_iter = find_nearby_parameter_grid(
        #     np.copy(np.squeeze(mcmc_output_dict['rho_0'])[ref_mcmc_id]), rho_grid
        # )

        beta_n_tar_ref_iter = np.copy(beta_tar_iter[nn + 1, :])
        sigma_n_ref_iter = np.copy(sigma_iter[nn + 1])
        rho_n_ref_iter = np.copy(rho_iter[nn + 1])

        _, z_vector_prob_iter_2_n, z_vector_iter_2_n = update_indicator_n(
            beta_0_tar_iter, beta_n_tar_ref_iter, sigma_0_iter, sigma_n_ref_iter, rho_0_iter, rho_n_ref_iter,
            source_data_n, signal_length_2, **kwargs
        )
        z_vector_iter_2[nn] = z_vector_iter_2_n
        z_vector_prob_iter_2[nn] = z_vector_prob_iter_2_n

    '''
    z_vector_iter_2 = np.array([0, 0, 0, 0, 0, 0])
    z_vector_prob_iter_2 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # print('z: {}'.format(z_vector_iter_2))
    '''

    accept_iter_ls.append(psi_tar_accept)
    accept_iter_ls.append(sigma_accept)
    # accept_iter_ls.append(rho_accept)
    accept_iter_ls.append(psi_0_ntar_accept)

    log_joint_prob = compute_joint_log_likelihood_cluster(beta_tar_iter, beta_0_ntar_iter, alpha_tar_iter,
                                                          alpha_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter, sigma_iter,
                                                          rho_iter, z_vector_iter_2, z_vector_prob_iter_2,
                                                          eigen_val_dict, source_data, new_data, N_total,
                                                          signal_length_2, source_name_ls, **kwargs)

    return [[beta_tar_iter, beta_0_ntar_iter, alpha_tar_iter, alpha_0_ntar_iter,
             psi_tar_iter, psi_0_ntar_iter, sigma_iter, rho_iter,
             z_vector_iter_2, z_vector_prob_iter_2], accept_iter_ls, log_joint_prob]


def compute_joint_log_likelihood_cluster(
        beta_tar_iter, beta_0_ntar_iter,
        alpha_tar_iter, alpha_0_ntar_iter, psi_tar_iter, psi_0_ntar_iter,
        sigma_iter, rho_iter, z_vector_iter, z_vector_prob_iter,
        eigen_val_dict, source_data, new_data, N_total, signal_length_2,
        source_name_ls=None, **kwargs
):
    mvn_N_plus_tar_obj = []
    eigen_val_tar = eigen_val_dict['target']
    eigen_val_ntar = eigen_val_dict['non-target']

    log_prior_N_plus = 0
    log_indicator = 0
    log_source_data = 0
    # log_new_data = 0

    psi_loc = kwargs['psi_loc']
    psi_scale = kwargs['psi_scale']
    # sigma_df = kwargs['sigma_df']
    sigma_loc = kwargs['sigma_loc']
    sigma_scale = kwargs['sigma_scale']

    # non-target-unique: alpha_ntar_0, psi_ntar_0 here
    log_prior_alpha_ntar_0 = mvn(
        mean=np.zeros_like(alpha_0_ntar_iter),
        cov=np.diag(eigen_val_ntar)).logpdf(x=alpha_0_ntar_iter)
    log_prior_psi_ntar_0 = lognorm(loc=psi_loc, s=psi_scale).logpdf(x=psi_0_ntar_iter)
    log_prior_N_plus = log_prior_N_plus + log_prior_alpha_ntar_0 + log_prior_psi_ntar_0

    # target-parameters here
    for n in range(N_total):
        # parameter priors
        log_prior_alpha_tar_n = mvn(
            mean=np.zeros_like(alpha_tar_iter[n, :]),
            cov=np.diag(eigen_val_tar)).logpdf(x=alpha_tar_iter[n, :])
        log_prior_psi_tar_n = lognorm(loc=psi_loc, s=psi_scale).logpdf(x=psi_tar_iter[n])
        log_prior_sigma_n = halfcauchy(loc=sigma_loc, scale=sigma_scale).logpdf(x=sigma_iter[n])
        # log_prior_rho_n = beta(a=rho_alpha, b=rho_beta).logpdf(x=rho_iter[n])
        # change it to discrete uniform, the same across rho_grid
        log_prior_N_plus = log_prior_N_plus + log_prior_alpha_tar_n + \
                           log_prior_psi_tar_n + log_prior_sigma_n

        cov_n = create_exponential_decay_cov_mat(sigma_iter[n] ** 2, rho_iter[n], signal_length_2)
        mvn_n_tar_obj = mvn(mean=beta_tar_iter[n, :], cov=cov_n)
        mvn_N_plus_tar_obj.append(mvn_n_tar_obj)

    # data likelihood contribution here
    # new subject, non-target data here
    cov_0 = create_exponential_decay_cov_mat(sigma_iter[0] ** 2, rho_iter[0], signal_length_2)
    mvn_0_ntar_obj = mvn(mean=beta_0_ntar_iter, cov=cov_0)
    new_data_ntar = new_data['non-target']
    log_data_0_ntar = np.sum(mvn_0_ntar_obj.logpdf(x=new_data_ntar))

    # new subject, target data here
    new_data_tar = new_data['target']
    log_data_0_tar = np.sum(mvn_N_plus_tar_obj[0].logpdf(x=new_data_tar))
    log_new_data = log_data_0_tar + log_data_0_ntar

    # source subject, target data here
    for nn in range(N_total - 1):
        if source_name_ls is None:
            subject_n_name = 'subject_{}'.format(nn + 1)
        else:
            subject_n_name = source_name_ls[nn + 1]

        source_data_n_tar = source_data[subject_n_name]['target']
        if z_vector_iter[nn] == 1:
            log_data_n_tar = np.sum(mvn_N_plus_tar_obj[0].logpdf(x=source_data_n_tar))
            log_indicator = log_indicator + np.log(z_vector_prob_iter[nn])
        else:
            log_data_n_tar = np.sum(mvn_N_plus_tar_obj[nn + 1].logpdf(x=source_data_n_tar))
            log_indicator = log_indicator - np.log(z_vector_prob_iter[nn])
        log_source_data = log_source_data + log_data_n_tar

    log_joint_prob = log_prior_N_plus + log_new_data + log_source_data + log_indicator

    return log_joint_prob


def update_alpha_n_tar(
        alpha_tar, psi_tar, sigma, rho, z_vector,
        eigen_fun_mat_tar, eigen_val_tar,
        source_data, new_data, N_total, n, signal_length_2,
        seq_source_size_2, scenario_dir,
        ref_mcmc_id, source_name_ls=None, xdawn_bool=False, **kwargs
):
    r"""
    :param xdawn_bool:
    :param alpha_tar:
    :param psi_tar:
    :param sigma:
    :param rho:
    :param z_vector: N-dim binary vector
    :param eigen_fun_mat_tar:
    :param eigen_val_tar:
    :param source_data:
    :param new_data:
    :param N_total:
    :param n: participant id, ranging from 0 to N
    :param signal_length_2:
    :param seq_source_size_2: integer, sequenze size for source participants
    :param source_name_ls: list of source participants' names
    :param scenario_dir: string
    :param ref_mcmc_id: integer, draw samples from MCMC samples of existing reference method
    :return:
    """

    if n == 0:
        size_n_tar = new_data['target'].shape[0]
        x_n_tar_sum = np.sum(new_data['target'], axis=0)

        lambda_alpha_n_tar = np.diag(1.0 / eigen_val_tar)
        if 'prior_mean_alpha_0_tar' in kwargs.keys():
            prior_mean_alpha_0_tar = kwargs['prior_mean_alpha_0_tar']
            eta_alpha_n_tar = lambda_alpha_n_tar @ prior_mean_alpha_0_tar
        else:
            eta_alpha_n_tar = np.zeros_like(eigen_val_tar)

        for nn in range(N_total-1):
            if z_vector[nn] == 1:
                if source_name_ls is None:
                    subject_nn_name = 'subject_{}'.format(nn + 1)
                else:
                    subject_nn_name = source_name_ls[nn+1]
                source_data_nn_tar = source_data[subject_nn_name]['target']
                size_n_tar = size_n_tar + source_data_nn_tar.shape[0]
                x_n_tar_sum = x_n_tar_sum + np.sum(source_data_nn_tar, axis=0)

        sigma_0 = sigma[n]
        rho_0 = rho[n]
        psi_0_tar = psi_tar[n]

        cov_n = create_exponential_decay_cov_mat(sigma_0 ** 2, rho_0, signal_length_2)
        cov_inv_n = compute_inverse_matrix(cov_n)
        eta_alpha_n_tar = np.matmul(lambda_alpha_n_tar, eta_alpha_n_tar) + \
                          psi_0_tar * eigen_fun_mat_tar.T @ cov_inv_n @ x_n_tar_sum
        lambda_alpha_n_tar = lambda_alpha_n_tar + size_n_tar * psi_0_tar ** 2 * \
                             eigen_fun_mat_tar.T @ cov_inv_n @ eigen_fun_mat_tar
        alpha_n_tar_sample = sample_normal_from_canonical_form(lambda_alpha_n_tar, eta_alpha_n_tar)

    else:
        # if z_vector[n - 1] == 1:
        #     alpha_n_tar_sample = mvn(mean=np.zeros_like(eigen_val_tar), cov=np.diag(1.0 / eigen_val_tar)).rvs(size=1)
        # else:
        if source_name_ls is None:
            mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                scenario_dir, n, seq_source_size_2)
            )
        else:
            if xdawn_bool:
                n_components = kwargs['n_components']
                letter_dim_sub = kwargs['letter_dim_sub']
                mcmc_output_dict = sio.loadmat(
                    '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                        scenario_dir, source_name_ls[n - 1], letter_dim_sub, n_components, seq_source_size_2)
                )
            else:
                select_channel_ids_str = kwargs['select_channel_ids_str']
                letter_dim_sub = kwargs['letter_dim_sub']
                mcmc_output_dict = sio.loadmat(
                    '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                        scenario_dir, source_name_ls[n - 1], letter_dim_sub, select_channel_ids_str, seq_source_size_2)
                )
        alpha_n_tar_sample = mcmc_output_dict['alpha_tar_0'][ref_mcmc_id, :]

    alpha_tar[n, :] = np.copy(alpha_n_tar_sample)

    return alpha_tar


def update_alpha_0_ntar(
        psi_0_ntar, sigma, rho,
        eigen_fun_mat_ntar, eigen_val_ntar,
        new_data, signal_length_2
):
    r"""
    :param psi_0_ntar:
    :param sigma:
    :param rho:
    :param eigen_fun_mat_ntar:
    :param eigen_val_ntar:
    :param new_data:
    :param signal_length_2:
    :return: Only involve non-target data of the new participant, no selection is performed w.r.t non-target data.
    """

    size_0_ntar = new_data['non-target'].shape[0]
    x_0_ntar_sum = np.sum(new_data['non-target'], axis=0)

    sigma_0 = sigma[0]
    rho_0 = rho[0]

    cov_0 = create_exponential_decay_cov_mat(sigma_0 ** 2, rho_0, signal_length_2)
    cov_inv_0 = compute_inverse_matrix(cov_0)

    lambda_alpha_0_ntar = np.diag(1.0 / eigen_val_ntar)
    lambda_alpha_0_ntar = lambda_alpha_0_ntar + size_0_ntar * psi_0_ntar ** 2 * \
                          eigen_fun_mat_ntar.T @ cov_inv_0 @ eigen_fun_mat_ntar
    eta_alpha_0_ntar = psi_0_ntar * eigen_fun_mat_ntar.T @ cov_inv_0 @ x_0_ntar_sum
    alpha_0_ntar_sample = sample_normal_from_canonical_form(lambda_alpha_0_ntar, eta_alpha_0_ntar)

    return alpha_0_ntar_sample


def update_psi_n_tar_RWMH(
        alpha_tar, psi_tar_old, sigma, rho, z_vector,
        eigen_fun_mat_tar, source_data, new_data, N_total, n, signal_length_2,
        seq_source_size_2, scenario_dir, ref_mcmc_id, step_size,
        source_name_ls=None, xdawn_bool=False, **kwargs
):
    accept_bool = 0
    ln_loc = kwargs['psi_loc']
    ln_scale = kwargs['psi_scale']

    if n == 0:
        # use random walk as the proposal distribution
        psi_n_tar_try = np.random.normal(loc=psi_tar_old[n], scale=step_size, size=1)
        psi_n_tar_old = psi_tar_old[n]

        alpha_n_tar = alpha_tar[n, :]
        sigma_n = sigma[n]
        rho_n = rho[n]

        if psi_n_tar_try > 0:
            # prior log-likelihood (use log-normal (0, 1))
            prior_log_prob_n_tar_try = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_try)
            prior_log_prob_n_tar_old = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_n_tar_old)

            mvn_n_tar_try = create_mvn_object(alpha_n_tar, eigen_fun_mat_tar, psi_n_tar_try, sigma_n, rho_n,
                                              signal_length_2)
            mvn_n_tar_old = create_mvn_object(alpha_n_tar, eigen_fun_mat_tar, psi_n_tar_old, sigma_n, rho_n,
                                              signal_length_2)

            new_data_tar = new_data['target']
            data_log_prob_n_tar_try = np.sum(mvn_n_tar_try.logpdf(new_data_tar))
            data_log_prob_n_tar_old = np.sum(mvn_n_tar_old.logpdf(new_data_tar))

            for nn in range(N_total-1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_nn_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_nn_name = source_name_ls[nn+1]
                    source_data_nn_tar = source_data[subject_nn_name]['target']
                    data_log_prob_n_tar_try = data_log_prob_n_tar_try + np.sum(mvn_n_tar_try.logpdf(source_data_nn_tar))
                    data_log_prob_n_tar_old = data_log_prob_n_tar_old + np.sum(mvn_n_tar_old.logpdf(source_data_nn_tar))

            alpha_log_ratio = data_log_prob_n_tar_try + prior_log_prob_n_tar_try - \
                              data_log_prob_n_tar_old - prior_log_prob_n_tar_old
            alpha_log_ratio = np.min([0, alpha_log_ratio[0]])

            if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
                psi_n_tar_old = np.copy(psi_n_tar_try)
                accept_bool = 1
    else:
        # if z_vector[n - 1] == 1:
        #     psi_n_tar_old = lognorm(loc=ln_loc, s=ln_scale).rvs(size=None)
        #     accept_bool = 0
        # else:
        if source_name_ls is None:
            mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                scenario_dir, n, seq_source_size_2)
            )
        else:
            if xdawn_bool:
                n_components = kwargs['n_components']
                letter_dim_sub = kwargs['letter_dim_sub']
                mcmc_output_dict = sio.loadmat(
                    '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                        scenario_dir, source_name_ls[n - 1], letter_dim_sub, n_components, seq_source_size_2)
                )
            else:
                select_channel_ids_str = kwargs['select_channel_ids_str']
                letter_dim_sub = kwargs['letter_dim_sub']
                mcmc_output_dict = sio.loadmat(
                    '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                        scenario_dir, source_name_ls[n - 1], letter_dim_sub, select_channel_ids_str, seq_source_size_2)
                )
        psi_n_tar_old = mcmc_output_dict['psi_tar_0'][0, ref_mcmc_id]
        accept_bool = 0

    psi_tar_old[n] = np.copy(psi_n_tar_old)

    return psi_tar_old, accept_bool


def update_psi_0_ntar_RWMH(
        alpha_0_ntar, psi_0_ntar_old, sigma, rho,
        eigen_fun_mat_ntar, new_data, signal_length_2,
        step_size=0.01, **kwargs
):
    # use random walk as the proposal distribution
    psi_0_ntar_try = np.random.normal(loc=psi_0_ntar_old, scale=step_size, size=None)
    accept_bool = 0
    ln_loc = kwargs['psi_loc']
    ln_scale = kwargs['psi_scale']

    if psi_0_ntar_try > 0:
        # prior log-likelihood (use log-normal (0, 1))
        prior_log_prob_0_ntar_try = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_0_ntar_try)
        prior_log_prob_0_ntar_old = lognorm(loc=ln_loc, s=ln_scale).logpdf(x=psi_0_ntar_old)

        # log-data-likelihood part
        data_log_prob_0_ntar_try = 0
        data_log_prob_0_ntar_old = 0

        sigma_0 = sigma[0]
        rho_0 = rho[0]

        # distribution object
        mvn_0_ntar_try = create_mvn_object(alpha_0_ntar, eigen_fun_mat_ntar, psi_0_ntar_try, sigma_0, rho_0,
                                           signal_length_2)
        mvn_0_ntar_old = create_mvn_object(alpha_0_ntar, eigen_fun_mat_ntar, psi_0_ntar_try, sigma_0, rho_0,
                                           signal_length_2)

        new_data_ntar = new_data['non-target']
        data_log_prob_0_ntar_try = data_log_prob_0_ntar_try + np.sum(mvn_0_ntar_try.logpdf(new_data_ntar))
        data_log_prob_0_ntar_old = data_log_prob_0_ntar_old + np.sum(mvn_0_ntar_old.logpdf(new_data_ntar))

        alpha_log_ratio = data_log_prob_0_ntar_try + prior_log_prob_0_ntar_try - \
                          data_log_prob_0_ntar_old - prior_log_prob_0_ntar_old

        alpha_log_ratio = np.min([0, alpha_log_ratio])
        # print('alpha_log_ratio is {}'.format(alpha_log_ratio))

        if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
            psi_0_ntar_old = psi_0_ntar_try
            accept_bool = 1

    return psi_0_ntar_old, accept_bool


def update_sigma_n_RWMH(
        beta_tar, beta_0_ntar, sigma_old, rho, z_vector,
        source_data, new_data, N_total, n, signal_length_2,
        seq_source_size_2, scenario_dir, ref_mcmc_id,
        step_size=0.1, source_name_ls=None, xdawn_bool=False, **kwargs
):
    # use random walk as the proposal distribution
    sigma_n_old = sigma_old[n]
    rho_n = rho[n]
    sigma_n_try = np.random.normal(loc=sigma_n_old, scale=step_size, size=None)
    accept_bool = 0
    cov_n_try = create_exponential_decay_cov_mat(sigma_n_try ** 2, rho_n, signal_length_2)
    sigma_loc = kwargs['sigma_loc']
    sigma_scale = kwargs['sigma_scale']

    if sigma_n_try > 0 and is_pos_def(cov_n_try):
        # prior log-likelihood (use half-cauchy (0, 1))
        prior_log_prob_n_try = halfcauchy(loc=sigma_loc, scale=sigma_scale).logpdf(x=sigma_n_try)
        prior_log_prob_n_old = halfcauchy(loc=sigma_loc, scale=sigma_scale).logpdf(x=sigma_n_old)
        beta_n_tar = beta_tar[n, :]

        if n == 0:
            # mvn distributional object
            mvn_n_tar_try = create_mvn_object_easy(beta_n_tar, sigma_n_try, rho_n, signal_length_2)
            mvn_n_tar_old = create_mvn_object_easy(beta_n_tar, sigma_n_old, rho_n, signal_length_2)
            # non-target for n=0 only
            mvn_n_ntar_try = create_mvn_object_easy(beta_0_ntar, sigma_n_try, rho_n, signal_length_2)
            mvn_n_ntar_old = create_mvn_object_easy(beta_0_ntar, sigma_n_old, rho_n, signal_length_2)

            new_data_tar = new_data['target']
            new_data_ntar = new_data['non-target']

            data_log_prob_n_try = np.sum(mvn_n_tar_try.logpdf(new_data_tar)) + \
                                  np.sum(mvn_n_ntar_try.logpdf(new_data_ntar))
            data_log_prob_n_old = np.sum(mvn_n_tar_old.logpdf(new_data_tar)) + \
                                  np.sum(mvn_n_ntar_old.logpdf(new_data_ntar))

            for nn in range(N_total - 1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_nn_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_nn_name = source_name_ls[nn + 1]
                    source_data_nn_tar = source_data[subject_nn_name]['target']
                    # only add contributions from source target data
                    data_log_prob_n_try = data_log_prob_n_try + np.sum(mvn_n_tar_try.logpdf(source_data_nn_tar))
                    data_log_prob_n_old = data_log_prob_n_old + np.sum(mvn_n_tar_old.logpdf(source_data_nn_tar))

            alpha_log_ratio = data_log_prob_n_try + prior_log_prob_n_try - data_log_prob_n_old - prior_log_prob_n_old
            alpha_log_ratio = np.min([0, alpha_log_ratio])

            if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
                sigma_n_old = sigma_n_try
                accept_bool = 1

        else:
            # if z_vector[n - 1] == 1:
            #     # sigma_n_old = sigma_old[0]
            #     # sigma_n_old = np.abs(stats.t(df=sigma_df).rvs(size=None))
            #     sigma_n_old = halfcauchy(loc=sigma_loc, scale=sigma_scale).rvs(size=None)
            #     accept_bool = 0
            # else:
            if source_name_ls is None:
                mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                    scenario_dir, n, seq_source_size_2)
                )
            else:
                if xdawn_bool:
                    n_components = kwargs['n_components']
                    letter_dim_sub = kwargs['letter_dim_sub']
                    mcmc_output_dict = sio.loadmat(
                        '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                            scenario_dir, source_name_ls[n - 1], letter_dim_sub, n_components, seq_source_size_2)
                    )
                else:
                    select_channel_ids_str = kwargs['select_channel_ids_str']
                    letter_dim_sub = kwargs['letter_dim_sub']
                    mcmc_output_dict = sio.loadmat(
                        '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                            scenario_dir, source_name_ls[n - 1], letter_dim_sub, select_channel_ids_str,
                            seq_source_size_2)
                    )
            sigma_n_old = mcmc_output_dict['sigma_0'][0, ref_mcmc_id]
            accept_bool = 0

        sigma_old[n] = np.copy(sigma_n_old)

    return sigma_old, accept_bool


def update_rho_n_tf_RWMH(
        beta_tar, beta_0_ntar, sigma, rho_old, z_vector,
        source_data, new_data, N_total, n, signal_length_2,
        seq_source_size_2, scenario_dir, ref_mcmc_id,
        step_size, source_name_ls=None, **kwargs
):
    hyper_lambda = 0.5
    rho_n_old = rho_old[n]
    accept_bool = 0
    rho_n_old_tf = np.log(rho_n_old / (1 - rho_n_old)) / hyper_lambda
    rho_n_try_tf = np.random.normal(loc=rho_n_old_tf, scale=step_size, size=None)
    rho_n_try = 1 / (1 + np.exp(-hyper_lambda * rho_n_try_tf))
    sigma_n = sigma[n]

    rho_tf_loc = kwargs['rho_loc']
    rho_tf_scale = kwargs['rho_scale']

    if n == 0:

        # prior log-likelihood (use uniform (0, 1))
        prior_log_prob_n_try = norm(loc=rho_tf_loc, scale=rho_tf_scale).logpdf(x=rho_n_try_tf)
        prior_log_prob_n_old = norm(loc=rho_tf_loc, scale=rho_tf_scale).logpdf(x=rho_n_old_tf)

        # mvn distributional objects
        beta_n_tar = beta_tar[n, :]

        mvn_n_tar_try = create_mvn_object_easy(beta_n_tar, sigma_n, rho_n_try, signal_length_2)
        mvn_n_tar_old = create_mvn_object_easy(beta_n_tar, sigma_n, rho_n_old, signal_length_2)
        # non-target for n=0 only
        mvn_n_ntar_try = create_mvn_object_easy(beta_0_ntar, sigma_n, rho_n_try, signal_length_2)
        mvn_n_ntar_old = create_mvn_object_easy(beta_0_ntar, sigma_n, rho_n_old, signal_length_2)

        new_data_tar = new_data['target']
        new_data_ntar = new_data['non-target']
        data_log_prob_n_try = np.sum(mvn_n_tar_try.logpdf(new_data_tar)) + \
                              np.sum(mvn_n_ntar_try.logpdf(new_data_ntar))
        data_log_prob_n_old = np.sum(mvn_n_tar_old.logpdf(new_data_tar)) + \
                              np.sum(mvn_n_ntar_old.logpdf(new_data_ntar))

        for nn in range(N_total - 1):
            if z_vector[nn] == 1:
                if source_name_ls is None:
                    subject_nn_name = 'subject_{}'.format(nn + 1)
                else:
                    subject_nn_name = source_name_ls[nn + 1]
                source_data_nn_tar = source_data[subject_nn_name]['target']
                data_log_prob_n_try = data_log_prob_n_try + np.sum(mvn_n_tar_try.logpdf(source_data_nn_tar))
                data_log_prob_n_old = data_log_prob_n_old + np.sum(mvn_n_tar_old.logpdf(source_data_nn_tar))

        alpha_log_ratio = data_log_prob_n_try + prior_log_prob_n_try - data_log_prob_n_old - prior_log_prob_n_old
        alpha_log_ratio = np.min([0, alpha_log_ratio])

        if np.random.uniform(0, 1, size=1) < np.exp(alpha_log_ratio):
            rho_n_old = rho_n_try
            accept_bool = 1

    else:

        # if z_vector[n - 1] == 1:
        #     rho_n_old_tf = norm(loc=rho_tf_loc, scale=rho_tf_scale).rvs(size=None)
        #     rho_n_old = 1 / (1 + np.exp(-hyper_lambda * rho_n_old_tf))
        #
        # else:
        mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
            scenario_dir, n, seq_source_size_2)
        )
        rho_n_old = mcmc_output_dict['rho_0'][0, ref_mcmc_id]

    rho_old[n] = np.copy(rho_n_old)

    return rho_old, accept_bool


def update_rho_n_discrete_uniform(
        beta_tar, beta_0_ntar, sigma, rho_old, z_vector,
        source_data, new_data, N_total, n, signal_length_2,
        seq_source_size_2, scenario_dir, ref_mcmc_id,
        source_name_ls=None, xdawn_bool=False, **kwargs
):
    rho_grid = kwargs['rho_grid']
    if n == 0:
        data_log_prob_n = []
        for rho_n_iter in rho_grid:
            beta_n_tar = beta_tar[n]
            sigma_n = sigma[n]

            mvn_n_tar_iter = create_mvn_object_easy(beta_n_tar, sigma_n, rho_n_iter, signal_length_2)
            mvn_n_ntar_iter = create_mvn_object_easy(beta_0_ntar, sigma_n, rho_n_iter, signal_length_2)

            new_data_tar = new_data['target']
            new_data_ntar = new_data['non-target']
            data_log_prob_n_iter = np.sum(mvn_n_tar_iter.logpdf(x=new_data_tar)) + \
                                   np.sum(mvn_n_ntar_iter.logpdf(x=new_data_ntar))
            for nn in range(N_total - 1):
                if z_vector[nn] == 1:
                    if source_name_ls is None:
                        subject_nn_name = 'subject_{}'.format(nn + 1)
                    else:
                        subject_nn_name = source_name_ls[nn + 1]
                    source_data_nn_tar = source_data[subject_nn_name]['target']
                    data_log_prob_n_iter = data_log_prob_n_iter + np.sum(mvn_n_tar_iter.logpdf(x=source_data_nn_tar))
            data_log_prob_n.append(data_log_prob_n_iter)
        data_log_prob_n = np.stack(data_log_prob_n, axis=0)
        data_prob_n = convert_log_prob_to_prob(data_log_prob_n)
        rho_n_sample = rho_grid[np.where(mtn(n=1, p=data_prob_n).rvs(size=None) == 1)[0][0]]
    else:
        if z_vector[n - 1] == 1:
            # rho_n_sample = rho_old[0]
            p_prior = np.ones(len(rho_grid)) / len(rho_grid)
            rho_n_sample = rho_grid[np.where(mtn(n=1, p=p_prior).rvs(size=None) == 1)[0][0]]

        else:
            if source_name_ls is None:
                mcmc_output_dict = sio.loadmat('{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(
                    scenario_dir, n, seq_source_size_2)
                )
            else:
                if xdawn_bool:
                    n_components = kwargs['n_components']
                    letter_dim_sub = kwargs['letter_dim_sub']
                    mcmc_output_dict = sio.loadmat(
                        '{}/{}/reference_numpyro_letter_{}_xdawn/channel_all_comp_{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(
                            scenario_dir, source_name_ls[n - 1], letter_dim_sub, n_components, seq_source_size_2)
                    )
                else:
                    select_channel_ids_str = kwargs['select_channel_ids_str']
                    letter_dim_sub = kwargs['letter_dim_sub']
                    mcmc_output_dict = sio.loadmat(
                        '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                            scenario_dir, source_name_ls[n - 1], letter_dim_sub, select_channel_ids_str,
                            seq_source_size_2)
                    )
            rho_n_sample = find_nearby_parameter_grid(
                np.copy(np.squeeze(mcmc_output_dict['rho_0'])[ref_mcmc_id]), rho_grid
            )
    rho_old[n] = np.copy(rho_n_sample)

    return rho_old


def update_indicator_n(
        beta_tar_0, beta_tar_n, sigma_0, sigma_n, rho_0, rho_n, source_data_n, signal_length_2, **kwargs
):
    """
   :param beta_tar_0:
   :param beta_tar_n:
   :param sigma_0:
   :param sigma_n:
   :param rho_0:
   :param rho_n:
   :param source_data_n:
   :param signal_length_2:
   :return: We update prob among source participants' data,
    so there are not non-target-related likelihood contribution.
   """
    approx_threshold = kwargs['approx_threshold']

    mvn_0_tar_obj = create_mvn_object_easy(beta_tar_0, sigma_0, rho_0, signal_length_2)
    log_prob_0_tar = np.sum(mvn_0_tar_obj.logpdf(source_data_n['target']))

    mvn_n_tar_obj = create_mvn_object_easy(beta_tar_n, sigma_n, rho_n, signal_length_2)
    log_prob_n_tar = np.sum(mvn_n_tar_obj.logpdf(source_data_n['target']))
    data_log_prob_vec = np.array([log_prob_0_tar, log_prob_n_tar])
    normalized_prob_n = convert_log_prob_to_prob(data_log_prob_vec)

    if np.abs(log_prob_0_tar - log_prob_n_tar) <= approx_threshold:
        normalized_prob_n = np.array([0.5, 0.5])
        # sample_n = 1
    # else:
    sample_n = np.random.binomial(1, normalized_prob_n[0], size=1)

    return data_log_prob_vec, normalized_prob_n[0], sample_n
