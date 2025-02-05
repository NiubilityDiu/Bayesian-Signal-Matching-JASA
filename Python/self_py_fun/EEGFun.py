from self_py_fun.Misc import *
rcp_unit_flash_num = 12
target_char_train = 'THE0QUICK0BROWN0FOX'
target_char_size = len(target_char_train)


def import_eeg_ML_dat(eeg_dat_dir):

    eeg_dat = sio.loadmat(eeg_dat_dir)
    eeg_signals = eeg_dat['Signal']
    eeg_code = np.squeeze(eeg_dat['Code'])
    eeg_type = np.squeeze(eeg_dat['Type'])

    return eeg_signals, eeg_type, eeg_code


def import_eeg_data_cluster_format(
        sub_name_ls, sub_new_name, letter_dim, channel_ids, signal_length,
        letter_dim_sub, seq_new_i, seq_source,
        parent_dir, dat_name_common, xdawn_bool,
        reshape_2d_bool=True, **kwargs
):
    E = 16
    N = len(sub_name_ls)
    if channel_ids is None:
        channel_ids = np.arange(E)
    E_sub = len(channel_ids)

    if letter_dim_sub is None:
        letter_dim_sub = letter_dim

    source_data = {}
    new_data = {}
    if xdawn_bool:
        n_components = kwargs['n_components']
    else:
        n_components = 0

    for n in range(N):
        subject_n_name = sub_name_ls[n]
        if subject_n_name in ['K154', 'K190']:
            seq_size = 20
        else:
            seq_size = 15
        sub_n_dir = '{}/{}'.format(parent_dir, subject_n_name)
        ref_method_name = 'reference_numpyro_letter_{}'.format(letter_dim_sub)
        if xdawn_bool:
            ref_method_name = '{}_xdawn'.format(ref_method_name)
        sub_n_ref_dir = '{}/{}'.format(sub_n_dir, ref_method_name)
        channel_name = 'channel_all_comp_{}'.format(n_components)

        if subject_n_name == sub_new_name:
            new_data = import_eeg_data_long_format(
                sub_new_name, letter_dim, seq_size, channel_ids,
                signal_length, seq_new_i, letter_dim_sub, parent_dir,
                dat_name_common, reshape_2d_bool
            )
            if xdawn_bool:
                xdawn_filter_new_dir = '{}/{}/xdawn_filter_train_seq_size_{}.mat'.format(
                    sub_n_ref_dir, channel_name, seq_new_i + 1
                )
            else:
                xdawn_filter_new_dir = None
                n_components = E_sub
            new_data_2, new_data_tar_size = produce_input_data_2(
                new_data, E_sub, signal_length, xdawn_filter_new_dir
            )
            new_data = process_n_component_dict(new_data_2, n_components, new_data_tar_size)
        else:
            source_data_n_raw = import_eeg_data_long_format(
                subject_n_name, letter_dim, seq_size, channel_ids,
                signal_length, seq_source - 1, letter_dim_sub, parent_dir,
                dat_name_common, reshape_2d_bool
            )

            if xdawn_bool:
                xdawn_filter_source_n_dir = '{}/{}/xdawn_filter_train_seq_size_{}.mat'.format(
                    sub_n_ref_dir, channel_name, seq_source
                )
            else:
                xdawn_filter_source_n_dir = None
                n_components = E_sub
            source_data_n_2, source_data_n_tar_size = produce_input_data_2(
                source_data_n_raw, E_sub, signal_length, xdawn_filter_source_n_dir
            )
            source_data[subject_n_name] = process_n_component_dict(source_data_n_2, n_components, source_data_n_tar_size)

    return source_data, new_data


def import_eeg_data_long_format(
        sub_new_name, letter_dim, seq_size, channel_ids, signal_length, seq_new_i, letter_dim_sub,
        parent_dir, dat_name_common, reshape_2d_bool=True
):
    E_total = 16
    E_sub = len(channel_ids)
    if letter_dim_sub is None:
        letter_dim_sub = letter_dim
    dat_n_dir = '{}/{}/{}_{}.mat'.format(parent_dir, sub_new_name, sub_new_name, dat_name_common)
    print(dat_n_dir)
    lower_seq_id = 1
    upper_seq_id = lower_seq_id + seq_new_i
    # We train lower_seq_id+1 to upper_seq_id+1 for each source participant
    signal_n_pseudo, type_n_pseudo, _ = import_eeg_train_data(
        dat_n_dir, upper_seq_id-1, channel_ids, E_total, signal_length, letter_dim, seq_size, reshape_2d_bool
    )
    signal_n_pseudo = np.reshape(
        signal_n_pseudo, [letter_dim, upper_seq_id, rcp_unit_flash_num, E_sub * signal_length]
    )
    type_n_pseudo = np.reshape(type_n_pseudo, [letter_dim, upper_seq_id, rcp_unit_flash_num])

    signal_n_pseudo = signal_n_pseudo[:letter_dim_sub, ...]
    type_n_pseudo = type_n_pseudo[:letter_dim_sub, :, :]

    sample_size_n_i_tar = letter_dim_sub * (seq_new_i+1) * 2
    sample_size_n_i_ntar = letter_dim_sub * (seq_new_i+1) * 10
    sample_size_n_i_total = letter_dim_sub * (seq_new_i+1) * rcp_unit_flash_num

    # let the notation be consistent
    signal_n_i = np.reshape(signal_n_pseudo[:, (lower_seq_id-1):, ...], [sample_size_n_i_total, E_sub * signal_length])
    type_n_i = np.reshape(type_n_pseudo[:, (lower_seq_id-1):, :], [sample_size_n_i_total])

    if reshape_2d_bool:
        new_data = {
            'target': np.reshape(
                signal_n_i[type_n_i == 1, :], [sample_size_n_i_tar, E_sub * signal_length]
            ),
            'non-target': np.reshape(
                signal_n_i[type_n_i != 1, :], [sample_size_n_i_ntar, E_sub * signal_length]
            )
        }
    else:
        new_data = {
            'target': np.reshape(
                signal_n_i[type_n_i == 1, ...], [sample_size_n_i_tar, E_sub, signal_length]
            ),
            'non-target': np.reshape(
                signal_n_i[type_n_i != 1, ...], [sample_size_n_i_ntar, E_sub, signal_length]
            )
        }

    return new_data


def import_eeg_train_data(
        train_data_dir, seq_i, channel_ids, E, signal_length,
        target_num_train, seq_size, reshape_2d_bool=True
):
    E_sub = len(channel_ids)
    # import training eeg_data
    eeg_train_data_dict = sio.loadmat(train_data_dir)
    signal_train = np.reshape(
        eeg_train_data_dict['Signal'], [target_num_train, seq_size, rcp_unit_flash_num, E, signal_length]
    )

    signal_train_i = signal_train[..., channel_ids, :]
    signal_train_i = signal_train_i[:, :(seq_i + 1), ...]

    sample_size_i = target_num_train * (seq_i + 1) * rcp_unit_flash_num
    signal_train_i = np.reshape(
        signal_train_i, [sample_size_i, E_sub, signal_length]
    )

    if reshape_2d_bool:
        signal_train_i = np.reshape(signal_train_i,[sample_size_i, E_sub * signal_length])

    label_train_i = np.reshape(
        np.reshape(eeg_train_data_dict['Type'],
                   [target_num_train, seq_size, rcp_unit_flash_num])[:, :(seq_i + 1), :],
        [sample_size_i]
    )
    code_train_i = np.reshape(
        np.reshape(eeg_train_data_dict['Code'],
                   [target_num_train, seq_size, rcp_unit_flash_num])[:, :(seq_i + 1), :],
        [sample_size_i]
    )

    return signal_train_i, label_train_i, code_train_i


def import_eeg_test_data(
        test_data_dir, channel_ids, E, signal_length, reshape_2d_bool=True
):
    # import training eeg_data
    eeg_test_data_dict = sio.loadmat(test_data_dir)
    signal_test = eeg_test_data_dict['Signal']
    total_flash_size, _ = signal_test.shape
    text_test = eeg_test_data_dict['Text']
    target_num_test = len(text_test)
    seq_size_test = int(total_flash_size / target_num_test / rcp_unit_flash_num)

    signal_test = np.reshape(signal_test, [total_flash_size, E, signal_length])[:, channel_ids, :]
    if reshape_2d_bool:
        E_sub = len(channel_ids)
        signal_test = np.reshape(signal_test, [total_flash_size, E_sub * signal_length])

    label_test = np.squeeze(eeg_test_data_dict['Type'], axis=1)
    code_test = np.squeeze(eeg_test_data_dict['Code'], axis=1)

    eeg_test_single_channel_dict = {
        'Signal': signal_test,
        'Type': label_test,
        'Code': code_test,
        'Text': text_test,
        'Text_size': target_num_test,
        'Seq_size': seq_size_test
    }

    return eeg_test_single_channel_dict

