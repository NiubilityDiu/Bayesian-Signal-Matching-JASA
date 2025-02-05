K_num_ids = c(106, 107, 108, 111, 112, 
              113, 114, 115, 117, 118, 
              119, 120, 121, 122, 123,
              143, 145, 146, 147, 151, 
              152, 154, 155, 156, 158, 
              159, 160, 166, 167, 171, 
              172, 177, 178, 179, 183, 
              184, 185, 190, 191, 212, 223)
K_sub_ids = paste('K', K_num_ids, sep="")
M_num_ids = c(131:136, 138:142, 144, 148, 149)
M_sub_ids = paste('M', M_num_ids, sep="")
sub_ids = c(K_sub_ids, M_sub_ids)
sub_size = length(sub_ids)


FRT_file_name_ls = list(
  'K112' = c('001_BCI_FRT'),
  'K122' = c('001_BCI_FRT'),
  'K154' = c('001_BCI_FRT', '002_BCI_FRT'),
  'K167' = c('001_BCI_FRT'),
  'K177' = c('001_BCI_FRT', '002_BCI_FRT'),
  'K212' = c('001_BCI_FRT'),
  'M131' = c('001_BCI_CPY', '002_BCI_CPY'),
  'M132' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M133' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M134' = c('001_BCI_CPY'),
  'M135' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M136' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M138' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M139' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M140' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M141' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M142' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M144' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M148' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY'),
  'M149' = c('001_BCI_CPY', '002_BCI_CPY', '003_BCI_CPY')
)

# RCP keyboard setup
# rcp_key_array = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
#                   'Y', 'Z', '1', '2', '3', '4', '5', 'SPEAK', '.', 'BS', '!', 'SPACE')
rcp_key_array = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                  'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
num_electrode = 16

channel_name_short = c('F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz')

sim_true_letters = strsplit('THE0QUICK0BROWN0FOX', '')[[1]]
sim_true_letter_len = length(sim_true_letters)

# sim_true_train_letters = strsplit('EEGBCI', '')[[1]]
sim_true_train_letters = strsplit(c('TTT'), '')[[1]]
sim_true_letter_train_len = length(sim_true_train_letters)


eeg_true_letters = strsplit('THE0QUICK0BROWN0FOX', '')[[1]]

eeg_reduced_source_ids = c(111, 113:115, 117, 120, 121, 123, 143,
                           146, 147, 151, 155, 156, 158, 166,
                           167, 171, 172, 177, 178, 179, 183, 184)
eeg_reduced_source_ids = paste('K', eeg_reduced_source_ids, sep='')
eeg_reduced_source_len = length(eeg_reduced_source_ids)

eeg_reduced_9_ids = c(143, 145, 146, 147, 151, 155, 171, 177, 178)
eeg_reduced_9_ids = paste('K', eeg_reduced_9_ids, sep='')
