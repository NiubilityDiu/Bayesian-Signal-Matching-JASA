from self_py_fun.SimFun import *
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


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
    var_ntar_name = 'kernel_ntar_var_new'
    var_ntar = numpyro.sample(var_ntar_name, dist.LogNormal(loc=0.0, scale=1.0))

    theta_ntar_name = 'theta_ntar_new'
    theta_ntar = numpyro.sample(
        theta_ntar_name, dist.MultivariateNormal(
            loc=jnp.zeros_like(eigen_val_dict['non-target']),
            covariance_matrix=jnp.diag(eigen_val_dict['non-target'])
        )
    )

    beta_ntar = var_ntar * jnp.matmul(eigen_fun_mat_dict['non-target'], theta_ntar)

    var_tar_name = 'kernel_tar_var_new'
    var_tar = numpyro.sample(var_tar_name, dist.LogNormal(loc=0.0, scale=1.0))

    theta_tar_name = 'theta_tar_new'
    theta_tar = numpyro.sample(
        theta_tar_name, dist.MultivariateNormal(
            loc=jnp.zeros_like(eigen_val_dict['target']),
            covariance_matrix=jnp.diag(eigen_val_dict['target'])
        )
    )
    beta_tar = var_tar * jnp.matmul(eigen_fun_mat_dict['target'], theta_tar)

    s_sq_name = 's_sq_new'
    s_sq = numpyro.sample(s_sq_name, dist.HalfCauchy(scale=1.0))
    rho_name = 'rho_new'
    rho = numpyro.sample(rho_name, dist.Beta(concentration1=1.0, concentration0=1.0))
    cov_mat = create_exponential_decay_cov_mat_jnp(s_sq, rho, signal_length)

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


def signal_new_sim_multi(
        input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None,
        channel_dim=None
):
    r"""
    :param input_points:
    :param eigen_val_dict:
    :param eigen_fun_mat_dict:
    :param input_data:
    :param channel_dim: int
    :return: This is a reference panel (restriction) for the new subject, try fitting GP separately
    """
    signal_length = len(input_points)
    var_ntar_ls = []
    theta_ntar_ls = []
    beta_ntar_ls = []
    beta_tar_ls = []
    s_sq_ls = []

    for e_id in range(channel_dim):
        var_ntar_e_name = 'kernel_ntar_var_new_{}'.format(e_id)
        var_ntar_e = numpyro.sample(var_ntar_e_name, dist.LogNormal(loc=0.0, scale=1.0))
        var_tar_e_name = 'kernel_tar_var_new_{}'.format(e_id)
        var_tar_e = numpyro.sample(var_tar_e_name, dist.LogNormal(loc=0.0, scale=1.0))

        theta_ntar_e_name = 'theta_ntar_new_{}'.format(e_id)
        theta_ntar_e = numpyro.sample(
            theta_ntar_e_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict['non-target']),
                covariance_matrix=jnp.diag(eigen_val_dict['non-target']))
        )
        theta_tar_e_name = 'theta_tar_new_{}'.format(e_id)
        theta_tar_e = numpyro.sample(
            theta_tar_e_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict['target']),
                covariance_matrix=jnp.diag(eigen_val_dict['target'])
            )
        )

        beta_ntar_e = var_ntar_e * jnp.matmul(eigen_fun_mat_dict['non-target'], theta_ntar_e)
        beta_tar_e = var_tar_e * jnp.matmul(eigen_fun_mat_dict['target'], theta_tar_e)

        s_sq_e_name = 's_sq_new_{}'.format(e_id)
        s_sq_e = numpyro.sample(s_sq_e_name, dist.HalfCauchy(scale=1.0))

        var_ntar_ls.append(var_ntar_e)
        theta_ntar_ls.append(theta_ntar_e)
        beta_ntar_ls.append(beta_ntar_e)
        beta_tar_ls.append(beta_tar_e)
        s_sq_ls.append(s_sq_e)

    rho_name = 'rho_new'
    rho = numpyro.sample(rho_name, dist.Beta(concentration0=1.0, concentration1=1.0))
    cov_rho_mat = create_exponential_decay_cov_mat_jnp(1.0, rho, signal_length)

    lambda_name = 'lambda_new'
    lambda_rv = numpyro.sample(lambda_name, dist.Beta(concentration0=1.0, concentration1=1.0))
    cov_lambda_mat = create_cs_cov_mat_jnp(s_sq_ls, lambda_rv, channel_dim)

    data_0_ntar_name = 'subject_0_ntar'
    new_0_ntar_dist = dist.MultivariateNormal(
        loc=jnp.reshape(jnp.stack(beta_ntar_ls), [channel_dim * signal_length]),
        covariance_matrix=jnp.kron(cov_lambda_mat, cov_rho_mat)
    )
    numpyro.sample(data_0_ntar_name, new_0_ntar_dist, obs=input_data['non-target'])

    data_0_tar_name = 'subject_0_tar'
    new_0_tar_dist = dist.MultivariateNormal(
        loc=jnp.reshape(jnp.stack(beta_tar_ls), [channel_dim * signal_length]),
        covariance_matrix=jnp.kron(cov_lambda_mat, cov_rho_mat)
    )
    numpyro.sample(data_0_tar_name, new_0_tar_dist, obs=input_data['target'])


def signal_integration_sim(
        N, K, input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None
):

    signal_length = len(input_points)
    var_ntar_dict = {}
    var_tar_dict = {}
    theta_ntar_dict = {}
    theta_tar_dict = {}

    beta_ntar_dict = {}
    beta_tar_dict = {}
    s_sq_dict = {}
    rho_dict = {}
    cov_mat_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)

        var_ntar_k_name = 'kernel_ntar_var_{}'.format(k)
        var_ntar_k = numpyro.sample(var_ntar_k_name, dist.LogNormal(loc=0.0, scale=1.0))
        # var_ntar_k = numpyro.sample(var_ntar_k_name, dist.TruncatedNormal(loc=0.0, scale=5.0, low=0.0, high=10.0))
        var_ntar_dict[var_ntar_k_name] = var_ntar_k

        theta_ntar_k_name = 'theta_ntar_{}'.format(k)
        theta_ntar_k = numpyro.sample(
            theta_ntar_k_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict[group_k_name]['non-target']),
                covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['non-target'])
            )
        )
        theta_ntar_dict[theta_ntar_k_name] = theta_ntar_k
        beta_ntar_k = var_ntar_k * jnp.matmul(eigen_fun_mat_dict[group_k_name]['non-target'], theta_ntar_k)
        beta_ntar_k_name = 'beta_ntar_{}'.format(k)
        beta_ntar_dict[beta_ntar_k_name] = beta_ntar_k

        var_tar_k_name = 'kernel_tar_var_{}'.format(k)
        var_tar_k = numpyro.sample(var_tar_k_name, dist.LogNormal(loc=0.0, scale=1.0))
        # var_tar_k = numpyro.sample(var_tar_k_name, dist.TruncatedNormal(loc=0.0, scale=5.0, low=0.0, high=10.0))
        var_tar_dict[var_tar_k_name] = var_tar_k

        theta_tar_k_name = 'theta_tar_{}'.format(k)
        theta_tar_k = numpyro.sample(
            theta_tar_k_name, dist.MultivariateNormal(
                loc=jnp.zeros_like(eigen_val_dict[group_k_name]['target']),
                covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['target'])
            )
        )
        theta_tar_dict[theta_tar_k_name] = theta_tar_k
        beta_tar_k = var_tar_k * jnp.matmul(eigen_fun_mat_dict[group_k_name]['target'], theta_tar_k)
        beta_tar_k_name = 'beta_tar_{}'.format(k)
        beta_tar_dict[beta_tar_k_name] = beta_tar_k

        s_sq_k_name = 's_sq_{}'.format(k)
        s_sq_k = numpyro.sample(s_sq_k_name, dist.HalfCauchy(scale=1.0))
        s_sq_dict[s_sq_k_name] = s_sq_k

        rho_k_name = 'rho_{}'.format(k)
        rho_k = numpyro.sample(rho_k_name, dist.Beta(concentration0=1.0, concentration1=1.0))
        rho_dict[rho_k_name] = rho_k
        cov_mat_k = create_exponential_decay_cov_mat_jnp(s_sq_k, rho_k, signal_length)
        '''
        cov_mat_k = s_sq_k * jnp.eye(signal_length)  # identity matrix
        '''
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

    for n in range(N-1):
        prob_n_name = 'prob_{}'.format(n+1)
        prob_n = numpyro.sample(prob_n_name, dist.Dirichlet(concentration=jnp.ones(K)))
        mixing_n = dist.Categorical(probs=prob_n)

        data_n_name = 'subject_{}'.format(n + 1)
        mixture_n_ntar_dist = dist.MixtureSameFamily(mixing_n, component_ntar_dist)
        numpyro.sample('{}_ntar'.format(data_n_name), mixture_n_ntar_dist, obs=input_data[data_n_name]['non-target'])

        mixture_n_tar_dist = dist.MixtureSameFamily(mixing_n, component_tar_dist)
        numpyro.sample('{}_tar'.format(data_n_name), mixture_n_tar_dist, obs=input_data[data_n_name]['target'])


def signal_integration_swap(mcmc_dict, N, K, arg_min_index):

    # swap the label between 0 and arg_min_index.
    mcmc_summary_dict = {
        'kernel_ntar_var_0': np.mean(mcmc_dict['kernel_ntar_var_{}'.format(arg_min_index)]),
        'kernel_tar_var_0': np.mean(mcmc_dict['kernel_tar_var_{}'.format(arg_min_index)]),
        's_sq_0': np.mean(mcmc_dict['s_sq_{}'.format(arg_min_index)]),
        'theta_ntar_0': np.mean(mcmc_dict['theta_ntar_{}'.format(arg_min_index)], axis=0),
        'theta_tar_0': np.mean(mcmc_dict['theta_tar_{}'.format(arg_min_index)], axis=0),

        'kernel_ntar_var_{}'.format(arg_min_index): np.mean(mcmc_dict['kernel_ntar_var_0']),
        'kernel_tar_var_{}'.format(arg_min_index): np.mean(mcmc_dict['kernel_tar_var_0']),
        's_sq_{}'.format(arg_min_index): np.mean(mcmc_dict['s_sq_0']),
        'theta_ntar_{}'.format(arg_min_index): np.mean(mcmc_dict['theta_ntar_0'], axis=0),
        'theta_tar_{}'.format(arg_min_index): np.mean(mcmc_dict['theta_tar_0'], axis=0)
    }

    for k in np.arange(1, K):
        if k != arg_min_index:
            mcmc_summary_dict['kernel_ntar_var_{}'.format(k)] = np.mean(mcmc_dict['kernel_ntar_var_{}'.format(k)])
            mcmc_summary_dict['kernel_tar_var_0'] = np.mean(mcmc_dict['kernel_tar_var_{}'.format(arg_min_index)])
            mcmc_summary_dict['s_sq_0'] = np.mean(mcmc_dict['s_sq_{}'.format(arg_min_index)])
            mcmc_summary_dict['theta_ntar_0'] = np.mean(mcmc_dict['theta_ntar_{}'.format(arg_min_index)], axis=0)
            mcmc_summary_dict['theta_tar_0'] = np.mean(mcmc_dict['theta_tar_{}'.format(arg_min_index)], axis=0)

    # efficient way to swap the indices of two values
    # https://stackoverflow.com/questions/14836228/is-there-a-standardized-method-to-swap-two-variables-in-python
    for n in np.arange(1, N):
        prob_n_name = 'prob_{}'.format(n)
        prob_n_mean = np.array(np.mean(mcmc_dict[prob_n_name], axis=0))
        # prob_n_mean_arg_min = np.copy(prob_n_mean[arg_min_index])
        # prob_n_mean[arg_min_index] = np.copy(prob_n_mean[0])
        # prob_n_mean[0] = np.copy(prob_n_mean_arg_min)
        prob_n_mean[arg_min_index], prob_n_mean[0] = prob_n_mean[0], prob_n_mean[arg_min_index]
        mcmc_summary_dict[prob_n_name] = prob_n_mean

    return mcmc_summary_dict


def signal_integration_sim_multi(
        N, K, input_points=None, eigen_val_dict=None,
        eigen_fun_mat_dict=None, input_data=None,
        channel_dim=None
):
    signal_length = len(input_points)

    beta_ntar_K_ls= []
    beta_tar_K_ls = []
    s_sq_K_ls = []
    rho_K_ls = []
    lambda_K_ls = []

    cov_rho_mat_K_ls = []
    cov_lambda_mat_K_ls = []

    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        var_ntar_k_ls = []
        var_tar_k_ls = []
        beta_ntar_k_ls = []
        beta_tar_k_ls = []
        s_sq_k_ls = []

        for e_id in range(channel_dim):
            var_ntar_k_e_name = 'kernel_ntar_var_{}_{}'.format(k, e_id)
            var_ntar_k_e = numpyro.sample(var_ntar_k_e_name, dist.LogNormal(loc=0.0, scale=1.0))
            var_ntar_k_ls.append(var_ntar_k_e)
            var_tar_k_e_name = 'kernel_tar_var_{}_{}'.format(k, e_id)
            var_tar_k_e = numpyro.sample(var_tar_k_e_name, dist.LogNormal(loc=0.0, scale=1.0))
            var_tar_k_ls.append(var_tar_k_e)

            theta_ntar_k_e_name = 'theta_ntar_{}_{}'.format(k, e_id)
            theta_ntar_k_e = numpyro.sample(
                theta_ntar_k_e_name, dist.MultivariateNormal(
                    loc=jnp.zeros_like(eigen_val_dict[group_k_name]['non-target']),
                    covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['non-target'])
                )
            )

            theta_tar_k_e_name = 'theta_tar_{}_{}'.format(k, e_id)
            theta_tar_k_e = numpyro.sample(
                theta_tar_k_e_name, dist.MultivariateNormal(
                    loc=jnp.zeros_like(eigen_val_dict[group_k_name]['target']),
                    covariance_matrix=jnp.diag(eigen_val_dict[group_k_name]['target'])
                )
            )

            beta_ntar_k_e = var_ntar_k_e * jnp.matmul(
                eigen_fun_mat_dict[group_k_name]['non-target'], theta_ntar_k_e
            )
            beta_tar_k_e = var_tar_k_e * jnp.matmul(
                eigen_fun_mat_dict[group_k_name]['target'], theta_tar_k_e
            )
            beta_ntar_k_ls.append(beta_ntar_k_e)
            beta_tar_k_ls.append(beta_tar_k_e)

            s_sq_k_e_name = 's_sq_{}_{}'.format(k, e_id)
            s_sq_k_e = numpyro.sample(s_sq_k_e_name, dist.HalfCauchy(scale=1.0))
            s_sq_k_ls.append(s_sq_k_e)

        beta_ntar_k_ls = jnp.stack(beta_ntar_k_ls, axis=0)
        beta_tar_k_ls = jnp.stack(beta_tar_k_ls, axis=0)
        beta_ntar_K_ls.append(beta_ntar_k_ls)
        beta_tar_K_ls.append(beta_tar_k_ls)
        s_sq_K_ls.append(s_sq_k_ls)

        rho_k_name = 'rho_{}'.format(k)
        rho_k = numpyro.sample(rho_k_name, dist.Beta(concentration0=1.0, concentration1=1.0))
        rho_K_ls.append(rho_k)
        cov_rho_mat_k = create_exponential_decay_cov_mat_jnp(1.0, rho_k, signal_length)
        cov_rho_mat_K_ls.append(cov_rho_mat_k)

        lambda_k_name = 'lambda_{}'.format(k)
        lambda_k = numpyro.sample(lambda_k_name, dist.Beta(concentration0=1.0, concentration1=1.0))
        lambda_K_ls.append(lambda_k)
        cov_lambda_mat_k = create_cs_cov_mat_jnp(s_sq_k_ls, lambda_k, channel_dim)
        cov_lambda_mat_K_ls.append(cov_lambda_mat_k)

    # initialize the component distribution of the mixture model by labels
    beta_tar_K_ls = jnp.reshape(jnp.stack(beta_tar_K_ls, axis=0), [K, channel_dim * signal_length])
    beta_ntar_K_ls = jnp.reshape(jnp.stack(beta_ntar_K_ls, axis=0), [K, channel_dim * signal_length])
    cov_mat_ls = []
    for k in range(K):
        cov_mat_ls.append(jnp.kron(cov_lambda_mat_K_ls[k], cov_rho_mat_K_ls[k]))
    cov_mat_ls = jnp.stack(cov_mat_ls, axis=0)

    component_ntar_dist = dist.MultivariateNormal(
        loc=beta_ntar_K_ls, covariance_matrix=cov_mat_ls
    )
    component_tar_dist = dist.MultivariateNormal(
        loc=beta_tar_K_ls, covariance_matrix=cov_mat_ls
    )

    # new subject component, non-target:
    data_0_ntar_name = 'subject_0_ntar'
    new_0_ntar_dist = dist.MultivariateNormal(
        loc=beta_ntar_K_ls[0, :],
        covariance_matrix=cov_mat_ls[0, ...]
    )
    numpyro.sample(data_0_ntar_name, new_0_ntar_dist, obs=input_data['subject_0']['non-target'])

    # new subject component, target:
    data_0_tar_name = 'subject_0_tar'
    new_0_tar_dist = dist.MultivariateNormal(
        loc=beta_tar_K_ls[0, :],
        covariance_matrix=cov_mat_ls[0, ...]
    )
    numpyro.sample(data_0_tar_name, new_0_tar_dist, obs=input_data['subject_0']['target'])

    # source subjects
    for n in range(N-1):
        prob_n_name = 'prob_{}'.format(n+1)
        prob_n = numpyro.sample(prob_n_name, dist.Dirichlet(concentration=jnp.ones(K)))
        mixing_n = dist.Categorical(probs=prob_n)

        data_n_name = 'subject_{}'.format(n + 1)
        mixture_n_ntar_dist = dist.MixtureSameFamily(mixing_n, component_ntar_dist)
        numpyro.sample('{}_ntar'.format(data_n_name), mixture_n_ntar_dist, obs=input_data[data_n_name]['non-target'])

        mixture_n_tar_dist = dist.MixtureSameFamily(mixing_n, component_tar_dist)
        numpyro.sample('{}_tar'.format(data_n_name), mixture_n_tar_dist, obs=input_data[data_n_name]['target'])


def signal_integration_swap_multi(mcmc_dict, N, K, channel_dim, arg_min_index):

    # swap the label between 0 and arg_min_index.
    mcmc_summary_dict = {}
    for e_id in range(channel_dim):
        kernel_ntar_var_0_e_name = 'kernel_ntar_var_0_{}'.format(e_id)
        mcmc_summary_dict[kernel_ntar_var_0_e_name] = np.mean(mcmc_dict[kernel_ntar_var_0_e_name])
        kernel_tar_var_0_e_name = 'kernel_tar_var_0_{}'.format(e_id)
        mcmc_summary_dict[kernel_tar_var_0_e_name] = np.mean(mcmc_dict[kernel_tar_var_0_e_name])

        theta_ntar_0_e_name = 'theta_ntar_0_{}'.format(e_id)
        mcmc_summary_dict[theta_ntar_0_e_name] = np.mean(mcmc_dict[theta_ntar_0_e_name], axis=0)
        theta_tar_0_e_name = 'theta_tar_0_{}'.format(e_id)
        mcmc_summary_dict[theta_tar_0_e_name] = np.mean(mcmc_dict[theta_tar_0_e_name], axis=0)

        s_sq_0_e_name = 's_sq_0_{}'.format(e_id)
        mcmc_summary_dict[s_sq_0_e_name] = np.mean(mcmc_dict[s_sq_0_e_name])

    rho_0_name = 'rho_0'
    mcmc_summary_dict[rho_0_name] = np.mean(mcmc_dict[rho_0_name])
    lambda_0_name = 'lambda_0'
    mcmc_summary_dict[lambda_0_name] = np.mean(mcmc_dict[lambda_0_name])

    for k in np.arange(1, K):
        if k != arg_min_index:
            for e_id in range(channel_dim):
                kernel_ntar_var_0_e_name = 'kernel_ntar_var_0_{}'.format(e_id)
                kernel_ntar_var_k_e_name = 'kernel_ntar_var_{}_{}'.format(k, e_id)
                mcmc_summary_dict[kernel_ntar_var_0_e_name] = np.mean(mcmc_dict[kernel_ntar_var_k_e_name])

                kernel_tar_var_0_e_name = 'kernel_tar_var_0_{}'.format(e_id)
                kernel_tar_var_k_e_name = 'kernel_tar_var_{}_{}'.format(k, e_id)
                mcmc_summary_dict[kernel_tar_var_0_e_name] = np.mean(mcmc_dict[kernel_tar_var_k_e_name])

                theta_ntar_0_e_name = 'theta_ntar_0_{}'.format(e_id)
                theta_ntar_k_e_name = 'theta_ntar_{}_{}'.format(k, e_id)
                mcmc_summary_dict[theta_ntar_0_e_name] = np.mean(mcmc_dict[theta_ntar_k_e_name], axis=0)

                theta_tar_0_e_name = 'theta_tar_0_{}'.format(e_id)
                theta_tar_k_e_name = 'theta_tar_{}_{}'.format(k, e_id)
                mcmc_summary_dict[theta_tar_0_e_name] = np.mean(mcmc_dict[theta_tar_k_e_name], axis=0)

                s_sq_0_e_name = 's_sq_0_{}'.format(e_id)
                s_sq_k_e_name = 's_sq_{}_{}'.format(k, e_id)
                mcmc_summary_dict[s_sq_0_e_name] = np.mean(mcmc_dict[s_sq_k_e_name])

            mcmc_summary_dict['rho_0'] = np.mean(mcmc_dict['rho_{}'.format(k)])
            mcmc_summary_dict['lambda_0'] = np.mean(mcmc_dict['lambda_{}'.format(k)])

    # efficient way to swap the indices of two values
    # https://stackoverflow.com/questions/14836228/is-there-a-standardized-method-to-swap-two-variables-in-python
    for n in np.arange(1, N):
        prob_n_name = 'prob_{}'.format(n)
        prob_n_mean = np.array(np.mean(mcmc_dict[prob_n_name], axis=0))
        # prob_n_mean_arg_min = np.copy(prob_n_mean[arg_min_index])
        # prob_n_mean[arg_min_index] = np.copy(prob_n_mean[0])
        # prob_n_mean[0] = np.copy(prob_n_mean_arg_min)
        prob_n_mean[arg_min_index], prob_n_mean[0] = prob_n_mean[0], prob_n_mean[arg_min_index]
        mcmc_summary_dict[prob_n_name] = prob_n_mean

    return mcmc_summary_dict


def signal_beta_summary(
        mcmc_dict: dict, eigen_fun_mat_dict: dict, suffix_name: str, q_low=0.05
):
    q_upp = 1 - q_low

    theta_ntar_name = 'theta_ntar_{}'.format(suffix_name)
    theta_ntar = mcmc_dict[theta_ntar_name]
    kernel_ntar_name = 'kernel_ntar_var_{}'.format(suffix_name)

    kernel_ntar = mcmc_dict[kernel_ntar_name]
    if len(kernel_ntar.shape) == 2:
        kernel_ntar = np.transpose(kernel_ntar)
    else:
        kernel_ntar = kernel_ntar[:, np.newaxis]

    beta_ntar = kernel_ntar * np.matmul(
        theta_ntar, np.transpose(eigen_fun_mat_dict['non-target'])
    )

    beta_ntar_mean = np.mean(beta_ntar, axis=0)
    beta_ntar_low = np.quantile(beta_ntar, q=q_low, axis=0)
    beta_ntar_upp = np.quantile(beta_ntar, q=q_upp, axis=0)

    beta_ntar_dict = {
        'beta_ntar_mean': beta_ntar_mean,
        'beta_ntar_low': beta_ntar_low,
        'beta_ntar_upp': beta_ntar_upp
    }

    theta_tar_name = 'theta_tar_{}'.format(suffix_name)
    theta_tar = mcmc_dict[theta_tar_name]
    kernel_tar_name = 'kernel_tar_var_{}'.format(suffix_name)

    kernel_tar = mcmc_dict[kernel_tar_name]
    if len(kernel_tar.shape) == 2:
        kernel_tar = np.transpose(kernel_tar)
    else:
        kernel_tar = kernel_tar[:, np.newaxis]

    beta_tar = kernel_tar * np.matmul(
        theta_tar, np.transpose(eigen_fun_mat_dict['target'])
    )
    beta_tar_mean = np.mean(beta_tar, axis=0)
    beta_tar_low = np.quantile(beta_tar, q=q_low, axis=0)
    beta_tar_upp = np.quantile(beta_tar, q=q_upp, axis=0)

    beta_tar_dict = {
        'beta_tar_mean': beta_tar_mean,
        'beta_tar_low': beta_tar_low,
        'beta_tar_upp': beta_tar_upp
    }

    return beta_tar_dict, beta_ntar_dict


def signal_integration_beta_summary(mcmc_dict, K, eigen_fun_mat_dict):

    beta_ntar_summary_dict = {}
    beta_tar_summary_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        beta_tar_k_dict, beta_ntar_k_dict = signal_beta_summary(mcmc_dict, eigen_fun_mat_dict[group_k_name], k)
        beta_tar_summary_dict[group_k_name] = beta_tar_k_dict
        beta_ntar_summary_dict[group_k_name] = beta_ntar_k_dict

    return beta_tar_summary_dict, beta_ntar_summary_dict


def signal_beta_multi_summary(
        mcmc_dict: dict, eigen_fun_mat_dict: dict, suffix_name: str, channel_dim: int, q_low=0.05
):
    q_upp = 1 - q_low

    beta_tar_ls = []
    beta_ntar_ls = []
    for channel_e in range(channel_dim):
        # ntarget
        theta_ntar_e_name = 'theta_ntar_{}_{}'.format(suffix_name, channel_e)
        theta_ntar_e = mcmc_dict[theta_ntar_e_name]
        kernel_ntar_e_name = 'kernel_ntar_var_{}_{}'.format(suffix_name, channel_e)
        kernel_ntar_e = mcmc_dict[kernel_ntar_e_name]

        if len(kernel_ntar_e.shape) == 2:
            kernel_ntar_e = np.transpose(kernel_ntar_e)
        else:
            kernel_ntar_e = kernel_ntar_e[:, np.newaxis]
        beta_ntar_e = kernel_ntar_e * np.matmul(
            theta_ntar_e, np.transpose(eigen_fun_mat_dict['non-target'])
        )
        beta_ntar_ls.append(beta_ntar_e)
        # target
        theta_tar_e_name = 'theta_tar_{}_{}'.format(suffix_name, channel_e)
        theta_tar_e = mcmc_dict[theta_tar_e_name]
        kernel_tar_e_name = 'kernel_tar_var_{}_{}'.format(suffix_name, channel_e)

        kernel_tar_e = mcmc_dict[kernel_tar_e_name]
        if len(kernel_tar_e.shape) == 2:
            kernel_tar_e = np.transpose(kernel_tar_e)
        else:
            kernel_tar_e = kernel_tar_e[:, np.newaxis]

        beta_tar_e = kernel_tar_e * np.matmul(
            theta_tar_e, np.transpose(eigen_fun_mat_dict['target'])
        )
        beta_tar_ls.append(beta_tar_e)

    beta_ntar_ls = np.stack(beta_ntar_ls, axis=1)
    beta_ntar_mean = np.mean(beta_ntar_ls, axis=0)
    beta_ntar_low = np.quantile(beta_ntar_ls, q=q_low, axis=0)
    beta_ntar_upp = np.quantile(beta_ntar_ls, q=q_upp, axis=0)
    beta_ntar_dict = {
        'beta_ntar_mean': beta_ntar_mean,
        'beta_ntar_low': beta_ntar_low,
        'beta_ntar_upp': beta_ntar_upp
    }

    beta_tar_ls = np.stack(beta_tar_ls, axis=1)
    beta_tar_mean = np.mean(beta_tar_ls, axis=0)
    beta_tar_low = np.quantile(beta_tar_ls, q=q_low, axis=0)
    beta_tar_upp = np.quantile(beta_tar_ls, q=q_upp, axis=0)

    beta_tar_dict = {
        'beta_tar_mean': beta_tar_mean,
        'beta_tar_low': beta_tar_low,
        'beta_tar_upp': beta_tar_upp
    }

    return beta_tar_dict, beta_ntar_dict


def signal_integration_beta_multi_summary(mcmc_dict, K, eigen_fun_mat_dict, channel_dim):

    beta_ntar_summary_dict = {}
    beta_tar_summary_dict = {}

    for k in range(K):
        group_k_name = 'group_{}'.format(k)
        beta_tar_k_dict, beta_ntar_k_dict = signal_beta_multi_summary(
            mcmc_dict, eigen_fun_mat_dict[group_k_name], str(k), channel_dim
        )
        beta_tar_summary_dict[group_k_name] = beta_tar_k_dict
        beta_ntar_summary_dict[group_k_name] = beta_ntar_k_dict

    return beta_tar_summary_dict, beta_ntar_summary_dict


''''# score-based, deprecated, will not be included in the final manuscript.
def score_integration_sim_N_2(K, q_mat=None, data_source=None, data_0=None):

    mu_ntar_0 = numpyro.sample('mu_ntar_0', dist.Normal(loc=0, scale=5))

    mu_diff_0_mean_prior = numpyro.sample(
        'mu_diff_0_mean_diff', dist.Normal(loc=0, scale=5)
    )
    mu_diff_0 = numpyro.sample('mu_diff_0', dist.TruncatedNormal(
        loc=mu_diff_0_mean_prior, scale=5, low=0, high=10
    ))
    # mu_diff_0 = numpyro.sample('mu_diff_0', dist.HalfNormal(scale=5))
    mu_tar_0 = mu_ntar_0 + mu_diff_0
    mu_vec_0 = jnp.array([mu_tar_0, mu_ntar_0])

    mu_ntar_1 = numpyro.sample('mu_ntar_1', dist.Normal(loc=0, scale=5))
    mu_diff_1 = numpyro.sample('mu_diff_1', dist.HalfNormal(scale=5))
    mu_tar_1 = mu_ntar_1 + mu_diff_1
    mu_vec_1 = jnp.array([mu_tar_1, mu_ntar_1])

    mu_vec = jnp.stack([mu_vec_0, mu_vec_1], axis=-1)
    mu_vec_mix = jnp.transpose(jnp.matmul(q_mat, mu_vec))
    mu_vec_0_long = jnp.squeeze(jnp.matmul(q_mat, mu_vec_0[:, jnp.newaxis]), axis=-1)

    s_sq_0 = numpyro.sample('s_sq_0', dist.HalfCauchy(scale=5))
    cov_mat_0 = s_sq_0 * jnp.eye(rcp_unit_flash_num)
    s_sq_1 = numpyro.sample('s_sq_1', dist.HalfCauchy(scale=5))
    cov_mat_1 = s_sq_1 * jnp.eye(rcp_unit_flash_num)
    cov_mat_mix = jnp.stack([cov_mat_0, cov_mat_1], axis=0)

    component_dist = dist.MultivariateNormal(
        loc=mu_vec_mix, covariance_matrix=cov_mat_mix
    )

    new_0_dist = dist.MultivariateNormal(
        loc=mu_vec_0_long, covariance_matrix=cov_mat_0
    )
    numpyro.sample('data_0', new_0_dist, obs=data_0)

    prob_1 = numpyro.sample('prob_1', dist.Dirichlet(concentration=jnp.ones(K)))
    mixing_1 = dist.Categorical(probs=prob_1)
    mixture_1_dist = dist.MixtureSameFamily(
        mixing_1, component_dist
    )
    numpyro.sample('data_1', mixture_1_dist, obs=data_source[0])


def score_integration_sim_N_3(K, q_mat=None, data_source=None, data_0=None):

    mu_ntar_0 = numpyro.sample('mu_ntar_0', dist.Normal(loc=0, scale=5))
    mu_diff_0 = numpyro.sample('mu_diff_0', dist.HalfNormal(scale=5))
    mu_tar_0 = mu_ntar_0 + mu_diff_0
    mu_vec_0 = jnp.array([mu_tar_0, mu_ntar_0])

    mu_ntar_1 = numpyro.sample('mu_ntar_1', dist.Normal(loc=0, scale=5))
    mu_diff_1 = numpyro.sample('mu_diff_1', dist.HalfNormal(scale=5))
    mu_tar_1 = mu_ntar_1 + mu_diff_1
    mu_vec_1 = jnp.array([mu_tar_1, mu_ntar_1])

    mu_vec = jnp.stack([mu_vec_0, mu_vec_1], axis=-1)
    mu_vec_mix = jnp.transpose(jnp.matmul(q_mat, mu_vec))
    mu_vec_0_long = jnp.squeeze(jnp.matmul(q_mat, mu_vec_0[:, jnp.newaxis]), axis=-1)

    s_sq_0 = numpyro.sample('s_sq_0', dist.HalfCauchy(5))
    cov_mat_0 = s_sq_0 * jnp.eye(rcp_unit_flash_num)
    s_sq_1 = numpyro.sample('s_sq_1', dist.HalfCauchy(5))
    cov_mat_1 = s_sq_1 * jnp.eye(rcp_unit_flash_num)
    cov_mat_mix = jnp.stack([cov_mat_0, cov_mat_1], axis=0)

    component_dist = dist.MultivariateNormal(
        loc=mu_vec_mix, covariance_matrix=cov_mat_mix
    )

    prob_1 = numpyro.sample('prob_1', dist.Dirichlet(concentration=jnp.ones(K)))
    prob_2 = numpyro.sample('prob_2', dist.Dirichlet(concentration=jnp.ones(K)))
    mixing_1 = dist.Categorical(probs=prob_1)
    mixing_2 = dist.Categorical(probs=prob_2)

    new_0_dist = dist.MultivariateNormal(
        loc=mu_vec_0_long, covariance_matrix=cov_mat_0
    )
    numpyro.sample('data_0', new_0_dist, obs=data_0)

    mixture_1_dist = dist.MixtureSameFamily(
        mixing_1, component_dist
    )
    numpyro.sample('data_1', mixture_1_dist, obs=data_source[0])

    mixture_2_dist = dist.MixtureSameFamily(
        mixing_2, component_dist
    )
    numpyro.sample('data_2', mixture_2_dist, obs=data_source[1])
'''
