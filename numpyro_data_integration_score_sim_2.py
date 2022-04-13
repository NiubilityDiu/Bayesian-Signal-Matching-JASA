import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from self_py_fun.SimFun import *
plt.style.use("bmh")
assert numpyro.__version__.startswith("0.9.0")

local_use = True
# local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')

target_char = 'T'
q_mat, q_mat_reparam = compute_q_matrix(target_char, sim_rcp_array)
N = 3
K = 2
np.random.seed(612)
if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)

prob_grp_0_subject_1 = 0.9
prob_grp_0_subject_2 = 0.9
# prob_grp_0_subject_1 = float(sys.argv[2])
# prob_grp_0_subject_2 = float(sys.argv[3])

mu_tar_group_1 = 0.9
# mu_tar_group_1 = float(sys.argv[4])

param_true_dict = {
    'target_char': target_char,
    'N': N,
    'seq_size': [10, 10, 10],
    'label_N': {'subject_0': np.array([1, 0]),
                'subject_1': np.array([prob_grp_0_subject_1, 1.0-prob_grp_0_subject_1]),
                'subject_2': np.array([prob_grp_0_subject_2, 1.0-prob_grp_0_subject_2])},
    'group_0': {'mu': np.array([[1.0], [0.0]]), 'sigma': 1},
    'group_1': {'mu': np.array([[mu_tar_group_1], [mu_tar_group_1-1.0]]), 'sigma': 1}
}

if prob_grp_0_subject_1 + prob_grp_0_subject_2 == 0.2:
    scenario_name = 'N_{}_K_{}_not_selected_overlap_{}'.format(N, K, mu_tar_group_1)
    print('NEITHER of subjects 1 and 2 are selected.')
elif prob_grp_0_subject_1 + prob_grp_0_subject_2 == 1.0:
    scenario_name = 'N_{}_K_{}_hybrid_overlap_{}'.format(N, K, mu_tar_group_1)
    print('One subject is selected, and another one is NOT.')
else:
    scenario_name = 'N_{}_K_{}_selected_overlap_{}'.format(N, K, mu_tar_group_1)
    print('Both subjects 1 and 2 are selected.')

scenario_name_dir = '{}/N_{}_K_{}'.format(parent_dir, N, K)
if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)
scenario_name_dir = '{}/{}'.format(scenario_name_dir, scenario_name)
if not os.path.exists(scenario_name_dir):
    os.mkdir(scenario_name_dir)

sim_data_dict = generate_simulated_data_score_based(param_true_dict, sim_rcp_array, scenario_name_dir)
data_sub0 = sim_data_dict['subject_0']['X']
data_source = [sim_data_dict['subject_1']['X'], sim_data_dict['subject_2']['X']]

prob_1_mean = []
prob_1_std = []
prob_2_mean = []
prob_2_std = []

mu_ntar_0_mean = []
mu_ntar_0_std = []
mu_tar_0_mean = []
mu_tar_0_std = []

mu_ntar_1_mean = []
mu_ntar_1_std = []
mu_tar_1_mean = []
mu_tar_1_std = []

seq_size = 10

for i in range(seq_size):
    print(i + 1)
    '''
    data_sub0_i = data_sub0[:(i + 1), :]
    nuts_kernel = NUTS(score_integration_sim_N_3)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, K, q_mat=q_mat, data_source=data_source, data_0=data_sub0_i,
             extra_fields=('potential_energy',))

    mcmc_dict = mcmc.get_samples()
    # save mcmc_dict
    mcmc_dict_name = 'mcmc_subject_0_seq_size_{}'.format(i + 1)
    mcmc_dict_dir = '{}/{}.mat'.format(scenario_name_dir, mcmc_dict_name)
    sio.savemat(mcmc_dict_dir, mcmc_dict)

    # save mcmc summary using sys.stdout
    summary_dict_name = 'summary_subject_0_seq_size_{}'.format(i + 1)
    summary_dict_dir = '{}/{}.txt'.format(scenario_name_dir, summary_dict_name)
    stdoutOrigin = sys.stdout
    sys.stdout = open(summary_dict_dir, "w")
    mcmc.print_summary()
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    '''

    # import mcmc_dict
    mcmc_dict_name = 'mcmc_subject_0_seq_size_{}'.format(i + 1)
    mcmc_dict_dir = '{}/{}.mat'.format(scenario_name_dir, mcmc_dict_name)
    mcmc_dict = sio.loadmat(mcmc_dict_dir)

    # save prob_1[0] value
    prob_1 = mcmc_dict['prob_1'][:, 0]
    prob_1_mean.append(np.mean(prob_1).item())
    prob_1_std.append(np.sqrt(np.var(prob_1)).item())

    # save prob_2[0] value
    prob_2 = mcmc_dict['prob_2'][:, 0]
    prob_2_mean.append(np.mean(prob_2).item())
    prob_2_std.append(np.sqrt(np.var(prob_2)).item())

    # save mu0, mu1 value
    mu_ntar_0 = mcmc_dict['mu_ntar_0']
    mu_diff_0 = mcmc_dict['mu_diff_0']
    mu_tar_0 = mu_ntar_0 + mu_diff_0
    mu_ntar_0_mean.append(np.mean(mu_ntar_0).item())
    mu_ntar_0_std.append(np.sqrt(np.var(mu_ntar_0)).item())
    mu_tar_0_mean.append(np.mean(mu_tar_0).item())
    mu_tar_0_std.append(np.sqrt(np.var(mu_tar_0)).item())

    mu_ntar_1 = mcmc_dict['mu_ntar_1']
    mu_diff_1 = mcmc_dict['mu_diff_1']
    mu_tar_1 = mu_ntar_1 + mu_diff_1
    mu_ntar_1_mean.append(np.mean(mu_ntar_1).item())
    mu_ntar_1_std.append(np.sqrt(np.var(mu_ntar_1)).item())
    mu_tar_1_mean.append(np.mean(mu_tar_1).item())
    mu_tar_1_std.append(np.sqrt(np.var(mu_tar_1)).item())

prob_1_mean = np.array(prob_1_mean)
prob_1_std = np.array(prob_1_std)
prob_2_mean = np.array(prob_2_mean)
prob_2_std = np.array(prob_2_std)

mu_ntar_0_mean = np.array(mu_ntar_0_mean)
mu_ntar_0_std = np.array(mu_ntar_0_std)
mu_tar_0_mean = np.array(mu_tar_0_mean)
mu_tar_0_std = np.array(mu_tar_0_std)

mu_ntar_1_mean = np.array(mu_ntar_1_mean)
mu_ntar_1_std = np.array(mu_ntar_1_std)
mu_tar_1_mean = np.array(mu_tar_1_mean)
mu_tar_1_std = np.array(mu_tar_1_std)


# errorbar plot: https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
fig1, ax1 = plt.subplots(2, 2, figsize=(18, 10))
ax1[0, 0].errorbar(
    np.arange(seq_size)+1, prob_1_mean, marker='o',
    yerr=prob_1_std, color='black', ecolor='gray', elinewidth=2
)
ax1[0, 0].set_ylim([0, 1])
ax1[0, 0].set_title(
    'prob_1[0], true value = {}'.format(prob_grp_0_subject_1),
    fontsize=10
)

ax1[0, 1].errorbar(
    np.arange(seq_size)+1, prob_2_mean, marker='o',
    yerr=prob_2_std, color='black', ecolor='gray', elinewidth=2
)
ax1[0, 1].set_ylim([0, 1])
ax1[0, 1].set_title(
    'prob_2[0], true value = {}'.format(prob_grp_0_subject_2),
    fontsize=10
)

ax1[1, 0].errorbar(
    np.arange(seq_size)+1, mu_ntar_0_mean, marker='o',
    yerr=mu_ntar_0_std, elinewidth=2, label='ntarget'
)
ax1[1, 0].errorbar(
    np.arange(seq_size)+1, mu_tar_0_mean, marker='o',
    yerr=mu_tar_0_std, elinewidth=2, label='target'
)
ax1[1, 0].legend(loc='best')
ax1[1, 0].hlines(
    xmin=1, xmax=10, y=np.array([
        param_true_dict['group_0']['mu'][0, 0],
        param_true_dict['group_0']['mu'][1, 0]]), linestyles='dashed', colors='black'
)
ax1[1, 0].set_xlabel('Sequence Number of Subject 0', fontsize=10)
ax1[1, 0].set_title(
    'mu_tar_0, mu_ntar_0, true values = {}, {}'.format(
        param_true_dict['group_0']['mu'][0, 0],
        param_true_dict['group_0']['mu'][1, 0]
    ), fontsize=10
)

ax1[1, 1].errorbar(
    np.arange(seq_size)+1, mu_ntar_1_mean, marker='o',
    yerr=mu_ntar_1_mean, elinewidth=2, label='ntarget'
)
ax1[1, 1].errorbar(
    np.arange(seq_size)+1, mu_tar_1_mean, marker='o',
    yerr=mu_tar_1_std, elinewidth=2, label='target'
)
ax1[1, 1].legend(loc='best')
ax1[1, 1].hlines(
    xmin=1, xmax=10, y=np.array([
        param_true_dict['group_1']['mu'][0, 0],
        param_true_dict['group_1']['mu'][1, 0]]), linestyles='dashed', colors='black'
)
ax1[1, 1].set_xlabel('Sequence Number of Subject 0', fontsize=10)
ax1[1, 1].set_title(
    'mu_tar_1, mu_ntar_1, true values = {}, {}'.format(
        param_true_dict['group_1']['mu'][0, 0],
        param_true_dict['group_1']['mu'][1, 0]
    ), fontsize=10
)
fig1.suptitle('Posterior summary', fontsize=15)
fig1.savefig('{}/plot_posterior_summary.png'.format(scenario_name_dir))

