import numpy as np
from pyriemann.spatialfilters import Xdawn
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt


def output_channel_ids(select_channel_ids):

    select_channel_ids = np.sort(select_channel_ids)
    E_sub = len(select_channel_ids)
    select_channel_ids_str = '_'.join((select_channel_ids + 1).astype('str').tolist())
    if len(select_channel_ids) == 16:
        select_channel_ids_str = 'all'
    return select_channel_ids, E_sub, select_channel_ids_str


def compute_inverse_matrix(input_matrix):

    input_chol_inv = np.linalg.inv(np.linalg.cholesky(input_matrix))
    input_matrix_inv = input_chol_inv.T @ input_chol_inv

    return input_matrix_inv


def find_nearby_parameter_grid(param_input, param_grid):
    param_grid = np.array(param_grid)
    rho_diff = np.abs(param_grid - param_input)
    arg_min_rho_diff = np.argmin(rho_diff)

    return param_grid[arg_min_rho_diff]


def pre_process_xdawn_train(input_signal, input_label, signal_length_2, xdawn_obj_dir,
                            E_sub=16, n_filters=2, reshape_3d_bool=True):
    xdawn_min = min(E_sub, n_filters)
    input_signal_size = input_signal.shape[0]
    xdawn_obj = Xdawn(nfilter=n_filters, estimator='scm')
    xdawn_obj.fit(input_signal, input_label)
    input_signal_xdawn = xdawn_obj.transform(input_signal)[:, :xdawn_min, :]
    input_signal_xdawn = np.reshape(input_signal_xdawn, [input_signal_size, xdawn_min * signal_length_2])
    if reshape_3d_bool:
        input_signal_xdawn = np.reshape(input_signal_xdawn, [input_signal_size, xdawn_min, signal_length_2])

    sio.savemat(
        xdawn_obj_dir, {
            'filter': xdawn_obj.filters_[:xdawn_min, :],
            'pattern': xdawn_obj.patterns_[:xdawn_min, :]
        }
    )

    return xdawn_obj, input_signal_xdawn


def produce_input_data_2(input_data, E_sub, signal_length_2, xdawn_filter_input_dir):
    input_data_X = np.concatenate([input_data['target'], input_data['non-target']], axis=0)
    input_data_tar_size = input_data['target'].shape[0]
    input_data_ntar_size = input_data['non-target'].shape[0]
    input_data_size = input_data_tar_size + input_data_ntar_size
    input_data_X = np.reshape(input_data_X, [input_data_size, E_sub, signal_length_2])
    if xdawn_filter_input_dir is None:
        input_data_2 = np.copy(input_data_X)
    else:
        xdawn_input_filter = sio.loadmat(xdawn_filter_input_dir)['filter']
        input_data_2 = xdawn_input_filter[np.newaxis, ...] @ input_data_X
    return input_data_2, input_data_tar_size


def process_n_component_dict(input_data_n_2, n_components, input_data_n_tar_size):
    if n_components == 1:
        input_data_n_2_dict = {
            'target': input_data_n_2[:input_data_n_tar_size, 0, :],
            'non-target': input_data_n_2[input_data_n_tar_size:, 0, :]
        }
    else:
        input_data_n_2_dict = {
            'target': input_data_n_2[:input_data_n_tar_size, ...],
            'non-target': input_data_n_2[input_data_n_tar_size:, ...]
        }

    return input_data_n_2_dict


def select_xdawn_ref_target_mcmc_dict_dir(
        source_name_ls, letter_dim_sub,
        seq_size, n, scenario_dir, **kwargs
):
    r"""
    :param source_name_ls:
    :param letter_dim_sub:
    :param seq_size:
    :param n:
    :param scenario_dir:
    :param kwargs:
    :return:
    """
    hyper_param_bool = kwargs['hyper_param_bool']
    n_components = kwargs['n_components']
    channel_name = 'channel_all_comp_{}'.format(n_components)
    ref_method_name = 'reference_numpyro_letter_{}_xdawn'.format(letter_dim_sub)
    ref_method_name_dir = '{}/{}/{}/{}'.format(
        scenario_dir, source_name_ls[n], ref_method_name, channel_name
    )

    mcmc_file_name = 'mcmc_seq_size_{}_reference_xdawn.mat'.format(seq_size)

    if hyper_param_bool:
        kernel_name = kwargs['kernel_name']
        mcmc_output_dict_dir = '{}/{}/{}'.format(ref_method_name_dir, kernel_name, mcmc_file_name)
    else:
        mcmc_output_dict_dir = '{}/{}'.format(ref_method_name_dir, mcmc_file_name)

    return mcmc_output_dict_dir


def select_ref_mcmc_output_dict_alternative(
        source_name_ls, letter_dim_sub, xdawn_bool, n, seq_size_val,
        scenario_dir, **kwargs
):
    if source_name_ls is None:
        mcmc_output_dict_dir = '{}/mcmc_sub_{}_seq_size_{}_reference.mat'.format(scenario_dir, n, seq_size_val)
    else:
        if xdawn_bool:
            mcmc_output_dict_dir = select_xdawn_ref_target_mcmc_dict_dir(
                source_name_ls, letter_dim_sub, seq_size_val,
                n, scenario_dir, **kwargs
            )
        else:
            select_channel_ids_str = kwargs['select_channel_ids_str']
            letter_dim_sub = kwargs['letter_dim_sub']
            mcmc_output_dict_dir = '{}/{}/reference_numpyro_letter_{}/channel_{}/mcmc_seq_size_{}_reference.mat'.format(
                    scenario_dir, source_name_ls[n - 1], letter_dim_sub, select_channel_ids_str,
                    seq_size_val
            )
    mcmc_output_dict = sio.loadmat(mcmc_output_dict_dir)

    return mcmc_output_dict


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def approx_inv_gamma_param_from_mu_and_var(mu_val, var_val):
    a = mu_val**2/var_val + 2.0
    scale = mu_val**3/var_val + mu_val
    return a, scale


def produce_ERP_and_z_vector_plots(
        N_total, x_time,
        B_tar_summary_dict, B_0_ntar_summary_dict, sigma_mean, z_vector_mean,
        signal_length, E_sub, seq_i,
        sub_name_reduce_ls, sub_new_cluster_dir_2, num_chain, **kwargs
):
    log_lhd_diff_approx_inner = kwargs['log_lhd_diff_approx']
    n_row, n_col = 5, 5
    if 'n_row' in kwargs.keys():
        n_row = kwargs['n_row']
    if 'n_col' in kwargs.keys():
        n_col = kwargs['n_col']
    y_low, y_upp = -2, 2
    if 'y_low' in kwargs.keys():
        y_low = kwargs['y_low']
    if 'y_upp' in kwargs.keys():
        y_upp = kwargs['y_upp']

    fig1, ax1 = plt.subplots(n_row, n_col, figsize=(45, 30))
    for row_i in range(n_row):
        for col_i in range(n_col):
            n = row_i * n_col + col_i
            if n < N_total:
                ax1[row_i, col_i].plot(x_time,
                                       np.reshape(B_tar_summary_dict['mean'][n, ...], [signal_length * E_sub]),
                                       label='Target', color='red')
                ax1[row_i, col_i].fill_between(
                    x_time, np.reshape(B_tar_summary_dict['low'][n, ...], [signal_length * E_sub]),
                    np.reshape(B_tar_summary_dict['upp'][n, ...], [signal_length * E_sub]), alpha=0.2,
                    label='Target', color='red'
                )
                ax1[row_i, col_i].set_ylim([y_low, y_upp])
                ax1[row_i, col_i].legend(loc='best')
                ax1[row_i, col_i].set_xlabel('Time (unit)')
                ax1[row_i, col_i].vlines(signal_length - 1, -3, 3)
                if n == 0:
                    ax1[row_i, col_i].set_title(
                        '{}, Sigma={:.2f},{:.2f}'.format(sub_name_reduce_ls[n], sigma_mean[n, 0], sigma_mean[n, 1]))
                else:
                    if z_vector_mean[n - 1] < 0.5:
                        ax1[row_i, col_i].set_title(
                            '{}, Sigma={:.2f},{:.2f}, Z={:.2f}'.format(sub_name_reduce_ls[n], sigma_mean[n, 0],
                                                                       sigma_mean[n, 1], z_vector_mean[n - 1]))
                    else:
                        ax1[row_i, col_i].set_title(
                            '{}, Sigma={:.2f},{:.2f}, Z={:.2f}'.format(sub_name_reduce_ls[n], sigma_mean[n, 0],
                                                                       sigma_mean[n, 1], z_vector_mean[n - 1]),
                            fontweight='bold')
            if n == 0:
                ax1[0, 0].plot(x_time, np.reshape(B_0_ntar_summary_dict['mean'], [signal_length * E_sub]),
                               label='Non-target', color='blue')
                ax1[0, 0].fill_between(
                    x_time, np.reshape(B_0_ntar_summary_dict['low'], [signal_length * E_sub]),
                    np.reshape(B_0_ntar_summary_dict['upp'], [signal_length * E_sub]),
                    alpha=0.2, label='Non-target', color='blue'
                )

        fig1.savefig('{}/plot_xdawn_seq_size_{}_{}_chains_log_lhd_diff_approx_{}.png'.format(
            sub_new_cluster_dir_2, seq_i + 1, num_chain, log_lhd_diff_approx_inner),
                     bbox_inches='tight')


def convert_device_array_to_numpy_array(input_dict: dict):
    input_dict_keys = input_dict.keys()
    for input_key in input_dict_keys:
        input_dict[input_key] = np.copy(np.array(input_dict[input_key]))
    return input_dict


def adjust_MH_step_size_random_walk(step_size_input, accept_rate_mean, low_bound=0.25, upper_bound=0.3):
    if accept_rate_mean > upper_bound:
        step_size_input = step_size_input * 1.1
    elif accept_rate_mean < low_bound:
        step_size_input = step_size_input * 0.9
    else:
        pass
    return step_size_input

