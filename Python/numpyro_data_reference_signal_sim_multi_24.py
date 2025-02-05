from self_py_fun.MCMCMultiFun import *
from self_py_fun.SimGlobal import *
plt.style.use("bmh")

E = 2
matrix_normal_bool = False
y_low, y_upp = -2, 2
parent_path_sim_dir  = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/SIM_files'.format(parent_path_sim_dir)
iter_num = 0
N_total = 24
K = 24
seq_i = 4

seq_size_ls = [seq_source_size for i in range(N_total)]

scenario_name = 'N_{}_K_{}_multi_xdawn_eeg'.format(N_total, K)
scenario_name_dir = '{}/{}/iter_{}'.format(parent_data_dir, scenario_name, iter_num)
sim_data_name = 'sim_dat'
sim_data_dir = '{}/{}.json'.format(
    scenario_name_dir, sim_data_name
)
ref_name_dir = '{}/reference_numpyro'.format(scenario_name_dir)
if not os.path.exists(ref_name_dir):
    os.mkdir(ref_name_dir)

# import sim_data
source_data, new_data = import_sim_data(
    sim_data_dir, seq_i, seq_size_ls, N_total, E, signal_length, matrix_normal_bool
)

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
    ref_name_dir, 0, seq_i + 1,
    y_low=y_low, y_upp=y_upp, num_samples=400
)

# source data
if seq_i == seq_size_ls[0] - 1:
    for n in range(N_total - 1):
        subject_n_name = 'subject_{}'.format(n + 1)
        print(subject_n_name)
        subject_n_data = source_data[subject_n_name]
        mcmc_sub_n_dict = numpyro_data_multi_reference_signal_sim_wrap_up(
            E, subject_n_data, index_x, eigen_val_dict, eigen_fun_mat_dict, n + 1,
            ref_name_dir, n + 1, seq_size_ls[n + 1],
            y_low=y_low, y_upp=y_upp, num_samples=400
        )
