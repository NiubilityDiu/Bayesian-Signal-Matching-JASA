from self_py_fun.MCMCMultiFun import *
from self_py_fun.EEGGlobal import *
import scipy.io as sio
from mne.viz import plot_topomap, plot_brain_colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use("bmh")


select_channel_ids = np.arange(E_total)
select_channel_ids, E_sub, select_channel_ids_str = output_channel_ids(select_channel_ids)
sensitivity_bool = True
z_threshold = 0.10

parent_path_eeg_local = '/Users/niubilitydiu/Desktop/BSM-Code-V2'
parent_data_dir = '{}/EEG_MATLAB_data/TRN_files'.format(parent_path_eeg_local)

if E_sub == 16:
    select_channel_ids_str = 'all'

sub_new_name = 'K151'
seq_i = 4
n_component = 2
log_lhd_diff_approx = 1.0
xdawn_min = min(n_component, E_sub)
x_time = np.arange(signal_length) / 32 * 1000

parent_data_sub_dir = '{}/{}'.format(parent_data_dir, sub_new_name)
if not os.path.exists('{}'.format(parent_data_sub_dir)):
    os.mkdir(parent_data_sub_dir)

if sensitivity_bool:
    # kernel hyper-parameter
    # Need add bottom caption.
    y_min = -2
    y_max = 1
    bottom_caption_color = 'black'
    length_vec = [0.35, 0.3, 0.25]
    gamma_vec = [1.25, 1.2, 1.15]
    sub_new_borrow_xdawn_dir = '{}/borrow_gibbs_letter_{}_reduced_xdawn/channel_{}_comp_{}'.format(
        parent_data_sub_dir, letter_dim_sub, select_channel_ids_str, n_component
    )
    sub_new_mixture_xdawn_dir = '{}/mixture_gibbs_letter_{}_reduced_xdawn/channel_{}_comp_{}'.format(
        parent_data_sub_dir, letter_dim_sub, select_channel_ids_str, n_component
    )
    mcmc_cluster_file_name = 'mcmc_seq_size_{}_cluster_xdawn_log_lhd_diff_approx_{}.mat'.format(
        seq_i+1, log_lhd_diff_approx
    )

    e = 0
    while e <= 1:
        fige0, axe0 = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
        for row_id, length_iter in enumerate(length_vec):
            for col_id, gamma_iter in enumerate(gamma_vec):
                length_ls = [[length_iter, length_iter-0.1]]
                gamma_val_ls = [[gamma_iter, gamma_iter]]
                eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
                    length_ls, gamma_val_ls, 1, signal_length
                )
                print('New subject name: {}'.format(sub_new_name))
                length_iter_2 = np.round(length_iter - 0.1, decimals=2)
                kernel_name = 'length_{}_{}_gamma_{}'.format(length_iter, length_iter_2, gamma_iter)
                # load borrow mcmc_dict
                mcmc_summary_borrow_xdawn_dir = '{}/{}/{}'.format(
                    sub_new_borrow_xdawn_dir, kernel_name, mcmc_cluster_file_name
                )
                mcmc_borrow_xdawn_dict = sio.loadmat(mcmc_summary_borrow_xdawn_dir, simplify_cells=True)
                mcmc_borrow_xdawn_dict = mcmc_borrow_xdawn_dict['chain_1']
                # produce the beta function plots
                beta_borrow_xdawn_tar_summary_dict, beta_borrow_xdawn_ntar_summary_dict = signal_B_multi_borrow_summary(
                    mcmc_borrow_xdawn_dict, str(0), 0.05
                )


                axe0[row_id, col_id].plot(x_time, beta_borrow_xdawn_ntar_summary_dict['mean'][e, :], label='Non-target', color='blue')
                axe0[row_id, col_id].plot(x_time, beta_borrow_xdawn_tar_summary_dict['mean'][e, :], label='Target', color='red')
                axe0[row_id, col_id].fill_between(x_time, beta_borrow_xdawn_ntar_summary_dict['low'][e, :],
                                                  beta_borrow_xdawn_ntar_summary_dict['upp'][e, :], alpha=0.2,
                                                  label='Non-target', color='blue')
                axe0[row_id, col_id].fill_between(x_time, beta_borrow_xdawn_tar_summary_dict['low'][e, :],
                                                  beta_borrow_xdawn_tar_summary_dict['upp'][e, :], alpha=0.2,
                                                  label='Target', color='red')
                axe0[row_id, col_id].set_ylim([y_min, y_max])
                axe0[row_id, col_id].legend(loc='lower right')
                # axe0[row_id, col_id].set_ylabel('Component {}'.format(e + 1))
                axe0[row_id, col_id].set_title('Length={},{}, Gamma={}'.format(length_iter, length_iter_2, gamma_iter), size=10)
        fige0.text(0.50, 0.08, 'Time (ms)', ha='center', size=15, color=bottom_caption_color)
        fige0.savefig('{}/{}_plot_xdawn_cluster_sensitivity_comparison_seq_size_{}_component_{}.png'.format(
            sub_new_mixture_xdawn_dir, sub_new_name, seq_i + 1, e+1), bbox_inches='tight', dpi=300)
        del fige0
        e = e + 1

else:
    y_min = -3
    y_max = 3
    # kernel hyper-parameter
    length_ls = [[0.3, 0.2]]
    gamma_val_ls = [[1.2, 1.2]]
    eigen_val_dict, eigen_fun_mat_dict = create_kernel_function(
        length_ls, gamma_val_ls, 1, signal_length
    )
    bottom_caption_color = 'black'
    print('New subject name: {}'.format(sub_new_name))

    # need to create a subfolder
    hyper_param_name = 'length_{}_{}_gamma_{}'.format(length_ls[0][0], length_ls[0][1], gamma_val_ls[0][0])
    # specify the kernel hyperparameter
    sub_new_borrow_xdawn_dir = '{}/borrow_gibbs_letter_{}_reduced_xdawn/channel_{}_comp_{}/{}'.format(
        parent_data_sub_dir, letter_dim_sub, select_channel_ids_str, n_component, hyper_param_name
    )
    # load borrow mcmc_dict
    mcmc_summary_borrow_xdawn_dir = '{}/mcmc_seq_size_{}_cluster_xdawn_log_lhd_diff_approx_{}.mat'.format(
        sub_new_borrow_xdawn_dir, seq_i + 1, log_lhd_diff_approx
    )
    mcmc_borrow_xdawn_dict = sio.loadmat(mcmc_summary_borrow_xdawn_dir, simplify_cells=True)
    mcmc_borrow_xdawn_dict = mcmc_borrow_xdawn_dict['chain_1']
    # produce the beta function plots
    beta_borrow_xdawn_tar_summary_dict, beta_borrow_xdawn_ntar_summary_dict = signal_B_multi_borrow_summary(
        mcmc_borrow_xdawn_dict, str(0), 0.05
    )
    z_vec_mcmc = mcmc_borrow_xdawn_dict['z_vector']
    z_vec_mean = np.mean(z_vec_mcmc, axis=0)

    # test spatial distribution data
    sub_new_ref_xdawn_dir = '{}/reference_numpyro_letter_{}_xdawn/channel_{}_comp_{}'.format(
        parent_data_sub_dir, letter_dim_sub, select_channel_ids_str, n_component
    )
    xdawn_filter_obj = sio.loadmat('{}/xdawn_filter_train_seq_size_{}.mat'.format(sub_new_ref_xdawn_dir, seq_i + 1))
    xdawn_spatial = xdawn_filter_obj['pattern']
    xdawn_filter = xdawn_filter_obj['filter']
    xdawn_spatial_min = np.round(np.min(xdawn_spatial, axis=0), decimals=1) - 0.1
    xdawn_spatial_max = np.round(np.max(xdawn_spatial, axis=0), decimals=1) + 0.1
    xdawn_spatial_mean = (xdawn_spatial_min + xdawn_spatial_max) / 2

    xdawn_filter_min = np.round(np.min(xdawn_filter, axis=0), decimals=1) - 0.1
    xdawn_filter_max = np.round(np.max(xdawn_filter, axis=0), decimals=1) + 0.1
    xdawn_filter_mean = (xdawn_filter_min + xdawn_filter_max) / 2

    cmap_option = 'Wistia'
    # z_vec_threshold_bool = np.sum((z_vec_mean >= z_threshold))
    fig0, ax0 = plt.subplots(nrows=xdawn_min, ncols=3, figsize=(12, 6))
    for e in range(xdawn_min):
        ax0[e, 0].plot(x_time, beta_borrow_xdawn_ntar_summary_dict['mean'][e, :], label='Non-target',
                       color='blue')
        ax0[e, 0].plot(x_time, beta_borrow_xdawn_tar_summary_dict['mean'][e, :], label='Target', color='red')
        ax0[e, 0].fill_between(x_time, beta_borrow_xdawn_ntar_summary_dict['low'][e, :],
                               beta_borrow_xdawn_ntar_summary_dict['upp'][e, :], alpha=0.2,
                               label='Non-target', color='blue')
        ax0[e, 0].fill_between(x_time, beta_borrow_xdawn_tar_summary_dict['low'][e, :],
                               beta_borrow_xdawn_tar_summary_dict['upp'][e, :], alpha=0.2,
                               label='Target', color='red')
        ax0[e, 0].set_ylim([y_min, y_max])
        # ax0[e, 0].legend(loc='best')
        ax0[e, 0].set_ylabel('Component {}'.format(e + 1))

        handles, labels = ax0[0, 0].get_legend_handles_labels()
        fig0.legend([(handles[0], handles[2]), (handles[1], handles[3])], ['Non-target', 'Target'],
                    loc=(0.08, 0.04), ncol=2, fontsize=10, framealpha=0.2)

        plot_topomap(data=xdawn_spatial[e, :], pos=position_2d, ch_type='eeg', cmap=cmap_option,
                     names=channel_name_short, size=4, show=False, axes=ax0[e, 1])
        divider_e_2 = make_axes_locatable(ax0[e, 1])
        cax_e_2 = divider_e_2.append_axes('right', size='5%', pad=0.5)
        cbar_e_2 = plot_brain_colorbar(cax_e_2,
                                       clim=dict(kind='value',
                                                 lims=[xdawn_spatial_min[e], xdawn_spatial_mean[e],
                                                       xdawn_spatial_max[e]]),
                                       orientation='vertical',
                                       colormap=cmap_option, label='')
        ax0[e, 1].set_title('')

        plot_topomap(data=xdawn_filter[e, :], pos=position_2d, ch_type='eeg', cmap=cmap_option,
                     names=channel_name_short, size=4, show=False, axes=ax0[e, 2])
        divider_e_3 = make_axes_locatable(ax0[e, 2])
        cax_e_3 = divider_e_3.append_axes('right', size='5%', pad=0.5)
        cbar_e_3 = plot_brain_colorbar(cax_e_3,
                                       clim=dict(kind='value',
                                                 lims=[xdawn_filter_min[e], xdawn_filter_mean[e],
                                                       xdawn_filter_max[e]]),
                                       orientation='vertical',
                                       colormap=cmap_option, label='')
        ax0[e, 2].set_title('')

    ax0[0, 0].set_title('BSM-Mixture', fontsize=11)
    fig0.text(0.5, 0.99, sub_new_name, ha='center', size=12, color=bottom_caption_color)
    fig0.text(0.24, 0.01, 'Time (ms)', ha='center', size=12, color=bottom_caption_color)
    fig0.text(0.49, 0.01, 'Spatial Pattern', ha='center', size=12, color=bottom_caption_color)
    fig0.text(0.77, 0.01, 'Spatial Filter', ha='center', size=12, color=bottom_caption_color)
    # plt.show()
    sub_new_mixture_xdawn_dir = '{}/mixture_gibbs_letter_{}_reduced_xdawn/channel_{}_comp_{}/{}'.format(
        parent_data_sub_dir, letter_dim_sub, select_channel_ids_str, n_component, hyper_param_name
    )
    fig0.savefig('{}/{}_plot_xdawn_bsm_mixture_seq_size_{}.png'.format(
        sub_new_mixture_xdawn_dir, sub_new_name, seq_i + 1), bbox_inches='tight', dpi=300)

    del fig0
