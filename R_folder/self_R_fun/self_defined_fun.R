library(R.matlab)
num_rep = 12

determine_row_column = function(target_char, rcp_key_arr) {
  target_char_num = which(rcp_key_arr == target_char)
  target_row = ifelse(target_char_num %% (num_rep/2) == 0, target_char_num %/% (num_rep/2), target_char_num %/% (num_rep/2) + 1)
  target_column = ifelse(target_char_num %% (num_rep/2) == 0, num_rep, target_char_num %% (num_rep/2) + (num_rep/2))
  return (list(row=target_row, column=target_column))
}


determine_letter = function(row_id, column_id, rcp_key_arr) {
  # row_id takes values among 1, ..., 6
  # column_id takes values among 7, ..., 12
  stopifnot(1 <= row_id, row_id <= num_rep/2, num_rep/2+1 <= column_id, column_id <= num_rep)
  key_id = (row_id - 1) * num_rep/2 + column_id - num_rep/2
  return (rcp_key_arr[key_id])
}


generate_stimulus_group_sequence = function(target_char, rcp_key_array) {
  target_output = determine_row_column(target_char, rcp_key_array)
  target_row = target_output$row
  target_column = target_output$column
  stimulus_code = sample.int(n=num_rep, size=num_rep, replace=F)
  stimulus_type = rep(-1, num_rep)
  stimulus_type[stimulus_code == target_row | stimulus_code == target_column] = 1
  
  return (list(code=stimulus_code, type=stimulus_type))
}


generate_simulated_data = function(target_chars, seq_num, rcp_key_array, param_ls) {
  target_char_size = length(target_chars)
  mu_tar = param_ls$mu_tar
  mu_ntar = param_ls$mu_ntar
  common_sd = param_ls$sd
  
  x_score = matrix(0, nrow=seq_num * num_rep, ncol=target_char_size)
  y_type = matrix(0, nrow=seq_num * num_rep, ncol=target_char_size)
  y_code = matrix(0, nrow=seq_num * num_rep, ncol=target_char_size)
  for (c in 1:target_char_size) {
    for (i in 1:seq_num) {
      y_output = generate_stimulus_group_sequence(target_chars[c], rcp_key_array)
      lower_i = (i-1)*num_rep+1
      upper_i = i * num_rep
      y_code[lower_i:upper_i,c] = y_output$code
      y_type[lower_i:upper_i,c] = y_output$type
      for (j in 1:num_rep) {
        x_score[lower_i+j-1,c] = ifelse(
          y_type[lower_i+j-1,c] == 1, rnorm(1, mu_tar, common_sd), rnorm(1, mu_ntar, common_sd)
        )
      }
    }
  }
  return(list(
    signal=x_score, type=y_type, 
    code=y_code, size=seq_num * num_rep * target_char_size
    ))
}


compute_binary_classification = function(df_input) {
  tab_df = table(df_input)
  total = signif((tab_df[1,1] + tab_df[2,2])/sum(tab_df), digits=2)
  se = tab_df[2,2] / sum(tab_df[2,])
  sp = tab_df[1,1] / sum(tab_df[1,])
  return (list(total=total, se=se, sp=sp))
}


obtain_char_level_prob = function(scores, stimulus_code, rcp_letter_table) {
  # This is for each super-trial (based on target character)
  stimulus_size = length(stimulus_code)
  seq_num = as.integer(stimulus_size / num_rep)
  
  single_score_row = single_score_column = matrix(0, nrow=num_rep/2, ncol=seq_num, byrow=F)
  
  for (j in 1:(num_rep/2)) {
    single_score_row[j,] = scores[stimulus_code == j] 
    single_score_column[j,] = scores[stimulus_code == j + num_rep/2]
  }
  # skip arg_max_single series
  cum_score_row = t(apply(single_score_row, 1, cumsum))
  cum_score_column = t(apply(single_score_column, 1, cumsum))
  
  arg_max_cum_row = apply(cum_score_row, 2, which.max)
  arg_max_cum_column = apply(cum_score_column, 2, which.max) + num_rep/2
  
  char_predict_seq = rep(0, seq_num)
  for (i in 1:seq_num) {
    char_predict_seq[i] = determine_letter(
      arg_max_cum_row[i], arg_max_cum_column[i], rcp_letter_table
    )
  }
  return (list(arg_max_cum_row = arg_max_cum_row, arg_max_cum_column = arg_max_cum_column,
               letter_cum=char_predict_seq))
}


remove_space = function(input_char) {
  input_char = as.character(input_char)
  letter_size = length(input_char)
  # print(letter_size)
  output_char = rep(0, letter_size)
  for (i in 1:letter_size) {
    if (input_char[i] == '<BS>') {
      output_char[i] = 'BS'
    } else if (input_char[i] == "    ") {
      output_char[i] = '_'
    } else {
      output_char[i] = gsub(' ', '', input_char[i])
    }
  }
  return (output_char)
}


self_cluster_quantitiy = function(input_df) {
  # within-cluster variability
  label_unique = sort(unique(input_df$label))
  ssw = rep(-1, length(label_unique))
  ssb = rep(-1, length(label_unique))
  
  for (label_id in label_unique) {
    percent_diff_w_id = input_df$percent_relative[input_df$label == label_id]
    ssw[label_id+1] = var(percent_diff_w_id)
    percent_diff_b_id = input_df$percent_relative[input_df$label != label_id]
    ssb[label_id+1] = var(percent_diff_b_id)
  }
  ssw_mean = mean(ssw)
  ssb_mean = mean(ssb)
  sstotal = ssw_mean + ssb_mean
  return (ssw_mean / sstotal)
}


self_method_cluster_quantity = function(input_list, sub_id, cluster_size) {
  # print(sub_id)
  input_list = input_list[[sub_id]]
  cluster_label = as.integer(input_list$Label)
  within_accuracy = input_list[[cluster_label+2]][[2]] / input_list[[cluster_label+2]][[1]]

  out_total = 0
  out_accuracy = 0
  for (cluster_id in 0:(cluster_size-1)) {
    if (cluster_id != cluster_label) {
      out_total = out_total + input_list[[cluster_id+2]][[1]]
      out_accuracy = out_accuracy + input_list[[cluster_id+2]][[2]]
    }
  }
  out_accuracy = out_accuracy / out_total
  output_df = data.frame(
    mean = c(within_accuracy, out_accuracy),
    sd = c(0, 0),
    subject = rep(sub_id, 2),
    cluster = c('Within', 'Out')
  )
  return (output_df)
}


self_method_single_cluster_quantity = function(exclude_ls, include_ls, sub_id) {
  # print(sub_id)
  exclude_list = exclude_ls[[sub_id]]
  include_list = include_ls[[sub_id]]
  cluster_label = 0
  exclude_accuracy = exclude_list[[cluster_label+2]][[2]] / exclude_list[[cluster_label+2]][[1]]
  include_accuracy = include_list[[cluster_label+2]][[2]] / include_list[[cluster_label+2]][[1]]
  
  exclude_total = exclude_list[[2]][[1]]
  exclude_accuracy = exclude_list[[2]][[2]]
  exclude_accuracy = exclude_accuracy / exclude_total
  
  include_total = include_list[[2]][[1]]
  include_accuracy = include_list[[2]][[2]]
  include_accuracy = include_accuracy / include_total
  
  output_df = data.frame(
    mean = c(include_accuracy, exclude_accuracy),
    sd = c(0, 0),
    subject = rep(sub_id, 2),
    cluster = c('Include', 'Exclude')
  )
  return (output_df)
}


# convert the data input from truncated_data.mat
convert_stan_input = function(eeg_mat, channel_dim) {
  eeg_signal = t(eeg_mat$Signal)   # (400, 3420)
  n_size = ncol(eeg_signal)
  l_spatial = 16  # by default
  l_temp = as.integer(nrow(eeg_signal) / l_spatial)
  eeg_signal = array(eeg_signal, dim=c(l_temp, l_spatial, n_size))
  # print(dim(eeg_signal))
  eeg_signal = aperm(eeg_signal, c(3, 1, 2))
  # print(dim(eeg_signal))
  
  eeg_type = eeg_mat$Type
  n_size_target = sum(eeg_type == 1)
  n_size_ntarget = sum(eeg_type != 1)
  eeg_type_scale = ifelse(eeg_type == 1, n_size_target / n_size, -n_size_ntarget /n_size)
  
  if (!is.null(channel_dim)) {
    print(paste('We will use channels ', channel_dim, sep=""))
    l_spatial = length(channel_dim)
    eeg_signal = eeg_signal[,,channel_dim]
  } else {
    print('We will use all 16 channels.')
  }
  
  m_mean_mat = matrix(0, nrow=l_temp, ncol=l_spatial)
  u_corr_temp_mat = diag(l_temp)
  v_corr_temp_mat = matrix(0.5, nrow=l_spatial, ncol=l_spatial)
  diag(v_corr_temp_mat) = 1
  return(stan_dat_ls = 
           list(l_temp = l_temp, l_spatial = l_spatial, 
              M_mean_mat = m_mean_mat, U_corr_temp_mat = u_corr_temp_mat, V_corr_temp_mat = v_corr_temp_mat,
              alpha_eta = 0.1, beta_eta = 0.1, n_size= n_size, 
              X_signal = eeg_signal, Y_label = eeg_type_scale[,1]
        ))
}


run_matlab_script_pass_param = function (fname, verbose = TRUE, desktop = FALSE, splash = FALSE, 
          display = FALSE, wait = TRUE, single_thread = FALSE, add_path = '', param = '', ...) {
  # stopifnot(file.exists(fname))
  matcmd = get_matlab(desktop = desktop, splash = splash, display = display, 
                      wait = wait, single_thread = single_thread)
  if (length(param)==0) {
    cmd = paste0(" \"", "try, run('", fname, "'); ", "catch err, disp(err.message); ", 
                 "exit(1); end; exit(0);", "\"")
  } else {
    cmd = paste0(" \"", "try, run('", fname, " ", param, "'); ", "catch err, disp(err.message); ", 
                 "exit(1); end; exit(0);", "\"")
  }
  
  if (length(add_path)==0) {
    cmd = paste0(matcmd, cmd)
  } else {
    add_path_2 = paste('addpath(genpath("', add_path, '"));', sep='')
    cmd = paste0(matcmd, add_path_2, cmd)
  }
  
  if (verbose) {
    message("Command run is:")
    message(cmd)
  }
  x <- system(cmd, wait = wait, ...)
  return(x)
}


import_data_obj = function(local_use, sub_name, data_type) {
  if (local_use) {
    parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  } else {
    parent_dir = '/home/mtianwen'
  }
  parent_data_dir = file.path(parent_dir, 'EEG_MATLAB_data', data_type, sub_name)
  data_dir = Sys.glob(file.path(parent_data_dir, '*Truncated_Data.mat*'))
  data_obj = readMat(data_dir)
  return (data_obj)
}


import_swlda_obj = function(local_use, sub_name, data_type, swlda_obj_name) {
  if (local_use) {
    parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode'
  } else {
    parent_dir = '/home/mtianwen'
  }
  parent_swlda_obj_dir = file.path(parent_dir, 'EEG_MATLAB_data', data_type, sub_name, 'swLDA')
  swlda_dir = Sys.glob(file.path(parent_swlda_obj_dir, paste(swlda_obj_name, '.mat', sep='')))
  # print(swlda_dir)
  swlda_obj = readMat(swlda_dir)
  return (swlda_obj)
}
 

compute_swlda_score = function(
  data_obj, swlda_obj, sample_size
) {
  
  b = as.numeric(swlda_obj$b)
  in_model = as.numeric(swlda_obj$in.model)
  b[in_model==0] = 0
  
  signal_mat = data_obj$Signal
  score = signal_mat %*% b
  type_indicator = data_obj$Type
  return (list(score=score[1:sample_size], type=type_indicator[1:sample_size], weight=b))
}


compute_swlda_score_summary = function(swlda_output) {
  
  swlda_score = swlda_output$score
  swlda_type = swlda_output$type
  
  target_mean = mean(swlda_score[swlda_type == 1])
  ntarget_mean = mean(swlda_score[swlda_type != 1])
  std_all = sd(swlda_score)
  d_prime = (target_mean - ntarget_mean) / std_all
  
  return (list(target_mean = target_mean,
               ntarget_mean = ntarget_mean,
               std = std_all,
               d_prime = d_prime))
}


compute_kl_divergence_normal = function(mu_new, sd_new, mu_ref, sd_ref) {
  d_kl = 0.5 * (sd_new^2 / sd_ref^2 - 1 + (mu_ref - mu_new)^2 / sd_ref^2 + 2*log(sd_ref/sd_new))
  return (dist = d_kl)
}

