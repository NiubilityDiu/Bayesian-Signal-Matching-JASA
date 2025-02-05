from self_py_fun.MCMCMultiFun import *
from self_py_fun.MCMCFun import *
from self_py_fun.EEGFun import *
from numpyro.infer import NUTS, MCMC
from self_py_fun.EEGGlobal import *
from jax import random
import seaborn as sns
import scipy.io as sio
plt.style.use("bmh")


# local_use = True
local_use = (sys.argv[1] == 'T' or sys.argv[1] == 'True')

if local_use:
    parent_dir = '{}/TRN_files'.format(parent_path_local)
    sub_new_name = 'K155'
    seq_i = 4
    n_components = 2
    length_ls = [[0.3, 0.2]]
    gamma_val_ls = [[1.2, 1.2]]
else:
    parent_dir = '{}/TRN_files'.format(parent_path_slurm)
    sub_new_name = sub_name_9_cohort_ls[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    seq_i = int(sys.argv[2])
    n_components = int(sys.argv[3])
    # sensitivity analysis check
    length_ls = [[float(sys.argv[4]), float(sys.argv[5])]]
    gamma_val_ls = [[float(sys.argv[6]), float(sys.argv[6])]]

lower_seq_id = 1
upper_seq_id = lower_seq_id + seq_i
seq_size = 15
if sub_new_name in ['K154', 'K190']:
    seq_size = 20

sub_new_dir = '{}/{}'.format(parent_dir, sub_new_name)
dat_name_common = '001_BCI_TRN_Truncated_Data_0.5_6'
sub_new_reference_dir = '{}/reference_numpyro_letter_{}_xdawn'.format(sub_new_dir, letter_dim_sub)
if not os.path.exists('{}'.format(sub_new_reference_dir)):
    os.mkdir(sub_new_reference_dir)

select_channel_ids_2 = np.arange(E_total)
E_sub = E_total

sub_new_reference_dir_2 = '{}/channel_all_comp_{}'.format(sub_new_reference_dir, n_components)
if not os.path.exists('{}'.format(sub_new_reference_dir_2)):
    os.mkdir(sub_new_reference_dir_2)

# import eeg_data
new_data_raw = import_eeg_data_long_format(
    sub_new_name, target_char_size, seq_size, select_channel_ids_2, signal_length,
    seq_i, letter_dim_sub, parent_dir, dat_name_common, reshape_2d_bool=True
)

# pre-process training set with xDAWN spatial filter
xdawn_min = min(E_sub, n_components)

new_data_X = np.concatenate([new_data_raw['target'], new_data_raw['non-target']], axis=0)
new_data_tar_size = new_data_raw['target'].shape[0]
new_data_ntar_size = new_data_raw['non-target'].shape[0]
new_data_y = np.concatenate([np.ones(new_data_tar_size), np.zeros(new_data_ntar_size)])
new_data_X = np.reshape(new_data_X, [new_data_tar_size + new_data_ntar_size, E_sub, signal_length])
xdawn_obj_dir = '{}/xdawn_filter_train_seq_size_{}.mat'.format(sub_new_reference_dir_2, seq_i+1)

xdawn_process_obj, new_data_X_xdawn = pre_process_xdawn_train(
    new_data_X, new_data_y, signal_length,
    xdawn_obj_dir, E_sub, n_components, reshape_3d_bool=False
)

'''
sns.heatmap(np.cov(new_data_X_xdawn[new_data_y==1, :25].T), cmap='YlGnBu')
plt.show()
sns.heatmap(np.cov(new_data_X_xdawn[new_data_y==0, :25].T), cmap='YlGnBu')
plt.show()
sns.heatmap(np.cov(new_data_X_xdawn[:, :25].T), cmap='YlGnBu')
plt.show()
'''

new_data = {
    'target': new_data_X_xdawn[:new_data_tar_size, :],
    'non-target': new_data_X_xdawn[new_data_tar_size:, :]
}

eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
    length_ls, gamma_val_ls, 1, signal_length
)

# semi-supervised clustering problem
rng_key_init = 2
rng_key_iter = random.PRNGKey(rng_key_init)
num_warmup = 2000
num_samples = 500

# sensitivity analysis check
kernel_name = 'length_{}_{}_gamma_{}'.format(length_ls[0][0], length_ls[0][1], gamma_val_ls[0][0])
print(kernel_name)
sub_new_reference_dir_2 = '{}/{}'.format(sub_new_reference_dir_2, kernel_name)
if not os.path.exists(sub_new_reference_dir_2):
    os.mkdir(sub_new_reference_dir_2)

# save the plots in Inference folder for better sorting purposes
inference_dir = '{}/Inference'.format(parent_dir)
if not os.path.exists(inference_dir):
    os.mkdir(inference_dir)
inference_dir_2 = '{}/xDAWN_descriptive'.format(inference_dir)
if not os.path.exists(inference_dir_2):
    os.mkdir(inference_dir_2)

# sensitivity analysis check
inference_dir_2 = '{}/{}'.format(inference_dir_2, kernel_name)
if not os.path.exists(inference_dir_2):
    os.mkdir(inference_dir_2)


if n_components > 1:
    # plot_new_data_multi_xdawn(
    #     new_data, new_data_X_xdawn, seq_i, new_data_tar_size, sub_new_name,
    #     n_components, sub_new_reference_dir_2, inference_dir_2
    # )

    nuts_kernel = NUTS(signal_new_sim_multi)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        rng_key_iter, input_points=index_x,
        eigen_val_dict=eigen_val_dict['group_0'],
        eigen_fun_mat_dict=eigen_fun_mat_dict['group_0'],
        input_data=new_data,
        E=xdawn_min,
        extra_fields=('potential_energy',)
    )

else:
    # plot_new_data_xdawn(
    #     new_data, new_data_X_xdawn, seq_i, new_data_tar_size, sub_new_name,
    #     n_components, sub_new_reference_dir_2, inference_dir_2
    # )

    nuts_kernel = NUTS(signal_new_sim)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        rng_key_iter, input_points=index_x,
        eigen_val_dict=eigen_val_dict['group_0'],
        eigen_fun_mat_dict=eigen_fun_mat_dict['group_0'],
        input_data=new_data,
        extra_fields=('potential_energy',)
    )

mcmc_iter_dict = mcmc.get_samples()
# convert it back to numpy array
mcmc_iter_dict = convert_device_array_to_numpy_array(mcmc_iter_dict)
# save mcmc_dict
# mcmc_iter_dict_dir = '{}/mcmc_seq_size_{}_reference_xdawn.npz'.format(sub_new_reference_dir_2, seq_i + 1)
# jaxnp.savez(mcmc_iter_dict_dir, mcmc_iter_dict)

mcmc_iter_dict_dir = '{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(sub_new_reference_dir_2, seq_i + 1)
sio.savemat(mcmc_iter_dict_dir, mcmc_iter_dict)

# save mcmc summary using sys.stdout
summary_dict_dir = '{}/summary_seq_size_{}_reference_xdawn.txt'.format(sub_new_reference_dir_2, seq_i + 1)
stdoutOrigin = sys.stdout
sys.stdout = open(summary_dict_dir, "w")
mcmc.print_summary()
sys.stdout.close()
sys.stdout = stdoutOrigin
mcmc.print_summary()

'''
# load mcmc_dict
mcmc_summary_dict_dir = '{}/mcmc_seq_size_{}_reference_xdawn.mat'.format(sub_new_reference_dir_2, seq_i + 1)
mcmc_iter_dict = sio.loadmat(mcmc_summary_dict_dir)
'''

# produce the beta function plots

if n_components > 1:
    plot_new_data_multi_xdawn_numpyro_reference_summary(
        mcmc_iter_dict, eigen_fun_mat_dict, seq_i, signal_length,
        xdawn_min, n_components, sub_new_reference_dir_2
    )

else:
    plot_new_data_xdawn_numpyro_reference_summary(
        mcmc_iter_dict, eigen_fun_mat_dict, seq_i,
        signal_length, sub_new_reference_dir_2
    )

