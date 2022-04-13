import numpy as np


sim_rcp_array = ['A', 'B', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'J', 'K', 'L',
                 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'T', 'U', 'V', 'W', 'X',
                 'Y', 'Z', '1', '2', '3', '4',
                 '5', '6', '7', '8', '9', '0']
parent_path_local = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/' \
                    'Dataset and Rcode/EEG_MATLAB_data'
parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'

channel_name_short = [
    'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz'
]

rcp_unit_flash_num = 12
rcp_char_size = 36
rcp_screen = np.reshape(np.arange(0, rcp_char_size), [6, 6]) + 1
stimulus_group_set = [
    rcp_screen[0, :], rcp_screen[1, :], rcp_screen[2, :], rcp_screen[3, :],rcp_screen[4, :], rcp_screen[5, :],
    rcp_screen[:, 0], rcp_screen[:, 1], rcp_screen[:, 2], rcp_screen[:, 3], rcp_screen[:, 4], rcp_screen[:, 5]
]
