import numpy as np

parent_path_local = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/' \
                    'Dataset and Rcode/EEG_MATLAB_data'
parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'

sim_rcp_array = ['A', 'B', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'J', 'K', 'L',
                 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'T', 'U', 'V', 'W', 'X',
                 'Y', 'Z', '1', '2', '3', '4',
                 '5', '6', '7', '8', '9', '0']

sub_name_ls = ['K106', 'K107', 'K108', 'K111', 'K112', 'K113', 'K114', 'K115',
               'K117', 'K118', 'K119', 'K120', 'K121', 'K122', 'K123', 'K143',
               'K145', 'K146', 'K147', 'K151', 'K152', 'K154', 'K155', 'K156',
               'K158', 'K159', 'K160', 'K166', 'K167', 'K171', 'K172', 'K177',
               'K178', 'K179', 'K183', 'K184', 'K185', 'K190', 'K191', 'K212', 'K223']
sub_throw_name_ls = ['K106', 'K107', 'K108', 'K112', 'K118',
                     'K119', 'K122', 'K145', 'K152', 'K154',
                     'K159', 'K160', 'K185', 'K190', 'K191',
                     'K212', 'K223']
sub_name_reduce_ls = []
for sub_new_name in sub_name_ls:
    if not sub_new_name in sub_throw_name_ls:
        sub_name_reduce_ls.append(sub_new_name)

sub_name_9_cohort_ls = ['K143', 'K145', 'K146', 'K147', 'K151', 'K155', 'K171', 'K177', 'K178']

# For smaller sample size:
# sub_top_10_swLDA_BSM_ref_ls = ["K183", "K114", "K117", "K151", "K178", "K121"]
# sub_top_10_swLDA_BSM_ref_ls = ["K114", "K117", "K121", "K151", "K178", "K183"]

# FRT files
FRT_file_name_dict = {
    'K112': ['001_BCI_FRT'],
    'K122': ['001_BCI_FRT'],
    'K154': ['001_BCI_FRT', '002_BCI_FRT'],
    'K167': ['001_BCI_FRT'],
    'K177': ['001_BCI_FRT', '002_BCI_FRT'],
    'K212': ['001_BCI_FRT'],
    'M131': ['001_BCI_CPY', '002_BCI_CPY'],
    'M132': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M133': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M134': ['001_BCI_CPY'],
    'M135': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M136': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M138': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M139': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M140': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M141': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M142': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M144': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M148': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'],
    'M149': ['001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY']
}

channel_ids = np.arange(16)
channel_name_short = [
    'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz'
]

position_x = np.array([-48, 0, 48, -87, -63, 0, 63, 87,
                       -59, 59, -48, 0, 48, -51, 51, 0])
position_y = np.array([59, 63, 59, 0, 0, 0, 0, 0,
                       -31, -31, -59, -63, -59, -71, -71, -87])
position_2d = np.stack([position_x, position_y], axis=1)


rcp_unit_flash_num = 12
rcp_char_size = 36
rcp_screen = np.reshape(np.arange(0, rcp_char_size), [6, 6]) + 1
stimulus_group_set = [
    rcp_screen[0, :], rcp_screen[1, :], rcp_screen[2, :], rcp_screen[3, :], rcp_screen[4, :], rcp_screen[5, :],
    rcp_screen[:, 0], rcp_screen[:, 1], rcp_screen[:, 2], rcp_screen[:, 3], rcp_screen[:, 4], rcp_screen[:, 5]
]

signal_length = 25
index_x = (np.arange(signal_length) - int(signal_length / 2)) / signal_length

target_num_train = 19
letter_dim_sub = 5  # THE_Q

# decision_rule_ls = ['NewOnly', 'Mixture']
decision_rule_ls = ['NewOnly']

E_total = 16
# select_channel_ids = np.array([15, 6]) - 1
# # select_channel_ids = np.arange(E_total)
# select_channel_ids = np.sort(select_channel_ids)
# E_sub = len(select_channel_ids)
# select_channel_ids_str = '_'.join((select_channel_ids + 1).astype('str').tolist())

length_super_ls = [
    [[0.3, 0.2]],
    [[0.3, 0.2]],
    [[0.3, 0.2]],
    [[0.25, 0.15]],
    [[0.25, 0.15]],
    [[0.25, 0.15]],
    [[0.35, 0.25]],
    [[0.35, 0.25]],
    [[0.35, 0.25]],
]
gamma_val_super_ls = [
    [[1.2, 1.2]],
    [[1.15, 1.15]],
    [[1.25, 1.25]],
    [[1.2, 1.2]],
    [[1.15, 1.15]],
    [[1.25, 1.25]],
    [[1.2, 1.2]],
    [[1.15, 1.15]],
    [[1.25, 1.25]]
]

