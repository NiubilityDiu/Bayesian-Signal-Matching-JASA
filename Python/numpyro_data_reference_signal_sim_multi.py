from self_py_fun.MCMCMultiFun import *
plt.style.use("bmh")

# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')
E = 2
matrix_normal_bool = False


if local_use:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_local)
    iter_num = 0
    N_total = 7
    K = 3
    prob_0_option_id = 1
    seq_i = 2

else:
    parent_dir = '{}/SIM_files/Chapter_3/numpyro_output'.format(parent_path_slurm)
    iter_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    N_total = int(sys.argv[2])
    K = int(sys.argv[3])
    prob_0_option_id = int(sys.argv[4])
    seq_i = int(sys.argv[5])  # take values from 0 to 9

seq_size_ls = [seq_source_size for i in range(N_total)]

scenario_name = 'N_{}_K_{}_multi_option_{}'.format(
    N_total, K, prob_0_option_id
)
scenario_name_dir = '{}/N_{}_K_{}/{}/iter_{}'.format(parent_dir, N_total, K, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)
ref_name_dir = '{}/reference_numpyro'.format(scenario_name_dir, K)
if not os.path.exists('{}'.format(ref_name_dir)):
    os.mkdir(ref_name_dir)

# import sim_data
source_data, new_data = import_sim_data(sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length, matrix_normal_bool)

# create the kernel matrix and related hyper-parameters
length_ls_2 = [[0.3, 0.2]]
gamma_val_ls_2 = [[1.2, 1.2]]
eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls_2, gamma_val_ls_2, 1, signal_length
)
index_x = (np.arange(signal_length) - int(signal_length / 2)) / signal_length

# new data first
mcmc_sub_0_dict = numpyro_data_multi_reference_signal_sim_wrap_up(
    E, new_data, index_x, eigen_val_dict, eigen_fun_mat_dict, 1,
    ref_name_dir, 0, seq_i + 1, y_low=-5, y_upp=5
)

# source data
if seq_i == seq_size_ls[0] - 1:
    for n in range(N_total - 1):
        subject_n_name = 'subject_{}'.format(n + 1)
        print(subject_n_name)
        subject_n_data = source_data[subject_n_name]
        mcmc_sub_n_dict = numpyro_data_multi_reference_signal_sim_wrap_up(
            E, subject_n_data, index_x, eigen_val_dict, eigen_fun_mat_dict, n + 1,
            ref_name_dir, n + 1, seq_size_ls[n + 1], y_low=-5, y_upp=5
        )
