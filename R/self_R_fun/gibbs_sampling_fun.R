# Apply the full Gibbs sampling procedure 
# to draw samples from posterior distributions
library(mvtnorm)
library(truncnorm)
library(invgamma)

# We assume the prior distribution is N(0, 1)
update_mu_source_ntar = function(
  mu_new_ntar, sigma_sq_source, delta_select, score_source, type_source, size_source_ntar
) {
  prior_sd = 10  # this is the prior sd of mu_source_ntar
  # lower_bound = -5; upper_bound = 5
  if (delta_select) {
    mu_source_ntar_gen = rnorm(1, mean=0, sd=prior_sd)
    # mu_source_ntar_gen = rtruncnorm(1, a=lower_bound, b=upper_bound, mean=0, sd=prior_sd)
    # mu_source_ntar_gen = mu_new_ntar
  } else {
    pres_post = size_source_ntar / sigma_sq_source + 1/prior_sd^2
    mu_post = (sum(score_source[type_source != 1]) / sigma_sq_source) / pres_post
    mu_source_ntar_gen = rnorm(1, mean=mu_post, sd=sqrt(1/pres_post))
    # mu_source_ntar_gen = rtruncnorm(1, a=lower_bound, b=upper_bound, mean=mu_post, sd=sqrt(1/pres_post))
  }
  return (mu_source_ntar = mu_source_ntar_gen)
}

update_mu_source_tar = function(
  mu_new_tar, mu_source_ntar, Delta_0_source, sigma_sq_source, delta_select, 
  score_source, type_source, size_source_tar
) {
  upper_bound = Inf
  prior_sd = 10 # this is the prior sd of mu_tar with truncated normal
  if (delta_select) {
    mu_source_tar_gen = rtruncnorm(
      1, a=mu_source_ntar, b=mu_source_ntar+upper_bound,
      mean=mu_source_ntar+Delta_0_source, sd=prior_sd
    )
    # mu_source_tar_gen = mu_new_tar
  } else {
    pres_post = size_source_tar / sigma_sq_source + 1/prior_sd^2
    mu_post = ((mu_source_ntar + Delta_0_source) / prior_sd^2 + 
                 sum(score_source[type_source == 1]) / sigma_sq_source) / pres_post
    mu_source_tar_gen = rtruncnorm(
      1, a=mu_source_ntar, b=mu_source_ntar+upper_bound, mean=mu_post, sd=sqrt(1/pres_post)
    )
  }
  return (mu_source_tar = mu_source_tar_gen)
}

update_mu_new_ntar = function(
  sigma_sq_new, delta_select_ls, source_data_ls, new_data_ls, source_name_vec, iter_id
) {
  prior_sd = 10 # this is the prior sd of mu_new_ntar
  pres_post = new_data_ls$size_ntar
  # lower_bound = -5; upper_bound = 5
  
  mu_numerator_post = sum(new_data_ls$score[new_data_ls$type != 1])
  for (source_name_iter in source_name_vec) {
    pres_post = pres_post + ifelse(
      delta_select_ls[[source_name_iter]][iter_id], source_data_ls[[source_name_iter]]$size_ntar, 0
    )
    mu_numerator_post = mu_numerator_post + ifelse(
      delta_select_ls[[source_name_iter]][iter_id], 
      sum(source_data_ls[[source_name_iter]]$score[source_data_ls[[source_name_iter]]$type != 1]), 0
    )
  }
  pres_post = pres_post / sigma_sq_new + 1/prior_sd^2
  mu_numerator_post = (mu_numerator_post / sigma_sq_new) / pres_post
  mu_new_ntar_gen = rnorm(1, mean=mu_numerator_post, sd=sqrt(1/pres_post))
  # mu_new_ntar_gen = rtruncnorm(
  #   1, a=lower_bound, b=upper_bound, mean=mu_numerator_post, sd=sqrt(1/pres_post)
  # )
  
  return (mu_new_ntar = mu_new_ntar_gen)
}

update_mu_new_tar = function(
  mu_new_ntar, Delta_0_new, sigma_sq_new, delta_select_ls, 
  source_data_ls, new_data_ls, source_name_vec, iter_id
) {
  upper_bound = Inf
  prior_sd = 10 # this is the prior sd of mu_new_tar with truncated normal
  pres_post = new_data_ls$size_tar
  mu_numerator_post = sum(new_data_ls$score[new_data_ls$type == 1])
  for (source_name_iter in source_name_vec) {
    pres_post = pres_post + ifelse(
      delta_select_ls[[source_name_iter]][iter_id], 
      source_data_ls[[source_name_iter]]$size_tar, 0
    )
    mu_numerator_post = mu_numerator_post + ifelse(
      delta_select_ls[[source_name_iter]][iter_id], 
      sum(source_data_ls[[source_name_iter]]$score[source_data_ls[[source_name_iter]]$type == 1]), 0
    )
  }
  pres_post = pres_post / sigma_sq_new + 1/prior_sd^2
  mu_numerator_post = (mu_numerator_post / sigma_sq_new + (mu_new_ntar+Delta_0_new) / prior_sd^2) / pres_post
  mu_new_tar_gen = rtruncnorm(1, a=mu_new_ntar, b=mu_new_ntar+upper_bound, mean=mu_numerator_post, sd=sqrt(1/pres_post))
  
  return (mu_new_tar = mu_new_tar_gen)
}

update_delta_select = function(
  mu_source_tar, mu_source_ntar, sigma_sq_source,
  mu_new_tar, mu_new_ntar, sigma_sq_new, 
  score_source, type_source, phi_0
) {
  log_L_new = sum(pnorm(score_source[type_source == 1], 
                        mean=mu_new_tar, sd=sqrt(sigma_sq_new), log=T)) + 
    sum(pnorm(score_source[type_source != 1], 
          mean=mu_new_ntar, sd=sqrt(sigma_sq_new), log=T))
  
  log_L_source = sum(pnorm(score_source[type_source == 1],
                       mean=mu_source_tar, sd=sqrt(sigma_sq_source), log=T)) +
    sum(pnorm(score_source[type_source != 1],
          mean=mu_source_ntar, sd=sqrt(sigma_sq_source), log=T))
  phi_post = 1 / (1 + exp(log_L_source + log(1-phi_0) - log_L_new - log(phi_0)))
  delta_select_gen = rbinom(1, 1, phi_post)
  return (list(delta_select = delta_select_gen, phi_post = phi_post))
}

update_sigma_sq_source_empirical = function(
  mu_tar, mu_ntar, score_source, type_source, size_tar, size_ntar
) {
  sq_sum = sum((score_source[type_source==1] - mu_tar)^2) +
    sum((score_source[type_source!=1] - mu_ntar)^2) 
  return (sq_sum / (size_tar + size_ntar - 1))
}

compute_log_lkd_source = function(
  mu_source_tar, mu_source_ntar, sigma_sq_source, score_source, type_source
) {
  log_lkd = sum(dnorm(score_source[type_source==1], mean=mu_source_tar, 
                  sd=sqrt(sigma_sq_source), log=T)) + 
    sum(dnorm(score_source[type_source!=1], mean=mu_source_ntar, 
              sd=sqrt(sigma_sq_source), log=T))
  return (log_lkd)
}

perform_sigma_sq_source_indep_MH = function(
  sigma_sq_new, mu_source_tar, mu_source_ntar, sigma_sq_source_old, delta_select_iter,
  score_source, type_source
) {
  t_df = 5
  if (delta_select_iter) {
    # log proposal distribution, use exp(2):
    exp_rate_param = 2
    sigma_sq_source_new = rexp(1, rate=exp_rate_param)
    log_prop_ratio = dexp(sigma_sq_source_old, rate=exp_rate_param, log=T) - 
      dexp(sigma_sq_source_new, rate=exp_rate_param, log=T)
    # log likelihood ratio
    log_lhd_ratio = compute_log_lkd_source(
      mu_source_tar, mu_source_ntar, sigma_sq_source_new, score_source, type_source
    ) - compute_log_lkd_source(
      mu_source_tar, mu_source_ntar, sigma_sq_source_old, score_source, type_source
    )
    # log prior ratio (half t-5 distribution)
    log_prior_ratio = dt(sigma_sq_source_new, df=t_df, log=T) - 
      dt(sigma_sq_source_old, df=t_df, log=T)
    log_alpha = min(0, log_prop_ratio + log_lhd_ratio + log_prior_ratio)
    
    if (log(runif(1)) < log_alpha) {
      sigma_sq_source_final = sigma_sq_source_new
    } else {
      sigma_sq_source_final = sigma_sq_source_old
    }
  } else {
    sigma_sq_source_final = rt(1, df=t_df, ncp=0)
    while (sigma_sq_source_final <= 0) {
      sigma_sq_source_final = rt(1, df=t_df, ncp=0)
    }
    # sigma_sq_source_final = sigma_sq_new
  }
  return (sigma_sq_source = sigma_sq_source_final)
}


compute_log_lkd_new = function(
  mu_new_tar, mu_new_ntar, sigma_sq_new, delta_select_ls, mcmc_iter,
  source_data_ls, new_data_ls, source_name_vec
) {
  log_lkd = sum(dnorm(new_data_ls$score[new_data_ls$type==1], mean=mu_new_tar, 
                      sd=sqrt(sigma_sq_new), log=T)) + 
    sum(dnorm(new_data_ls$score[new_data_ls$type!=1], mean=mu_new_ntar, 
              sd=sqrt(sigma_sq_new), log=T))
  for (source_name in source_name_vec) {
    log_lkd_name = ifelse(
      delta_select_ls[[source_name]][mcmc_iter], 
      sum(dnorm(source_data_ls[[source_name]]$score[source_data_ls[[source_name]]$type==1], mean=mu_new_tar, 
                sd=sqrt(sigma_sq_new), log=T)) + 
        sum(dnorm(source_data_ls[[source_name]]$score[source_data_ls[[source_name]]$type!=1], mean=mu_new_ntar, 
                  sd=sqrt(sigma_sq_new), log=T)), 0
    )
    log_lkd = log_lkd + log_lkd_name
  }
  return (log_lkd)
}

perform_sigma_sq_new_indep_MH = function(
  mu_new_tar, mu_new_ntar, sigma_sq_new_old, delta_select_ls, mcmc_iter,
  source_data_ls, new_data_ls, source_name_vec
) {
  # log proposal distribution, use exp(2):
  exp_rate_param = 2
  sigma_sq_new_new = rexp(1, rate=exp_rate_param)
  log_prop_ratio = dexp(sigma_sq_new_old, rate=exp_rate_param, log=T) - 
    dexp(sigma_sq_new_new, rate=exp_rate_param, log=T)
  # log likelihood ratio
  log_lhd_ratio = compute_log_lkd_new(
    mu_new_tar, mu_new_ntar, sigma_sq_new_new, delta_select_ls, mcmc_iter, 
    source_data_ls, new_data_ls, source_name_vec
  ) - compute_log_lkd_new(
    mu_new_tar, mu_new_ntar, sigma_sq_new_old, delta_select_ls, mcmc_iter, 
    source_data_ls, new_data_ls, source_name_vec
  )
  # log prior ratio (half t-5 distribution)
  t_df = 5
  log_prior_ratio = dt(sigma_sq_new_new, df=t_df, log=T) - 
    dt(sigma_sq_new_old, df=t_df, log=T)
  log_alpha = min(0, log_prop_ratio + log_lhd_ratio + log_prior_ratio)
  
  if (log(runif(1)) < log_alpha) {
    sigma_sq_new_final = sigma_sq_new_new
  } else {
    sigma_sq_new_final = sigma_sq_new_old
  }
  return (sigma_sq_new = sigma_sq_new_final)
}

perform_Delta_0_random_walk = function(
  mu_tar, mu_ntar, Delta_0_old, walk_size=0.5
) {
  prior_sd = 10  # this is the prior sd of truncated normal
  Delta_0_new = rnorm(1, mean=Delta_0_old, sd=walk_size)
  # log proposal ratio, use normal(0, 0.25)
  log_prop_ratio = dnorm(Delta_0_old - Delta_0_new, mean=0, sd=walk_size, log=T) - 
    dnorm(Delta_0_new - Delta_0_old, mean=0, sd=walk_size, log=T)
  # log likelihood ratio (truncated normal likelihood)
  log_lhd_ratio = dnorm(x=mu_tar-mu_ntar-Delta_0_new, mean=0, sd=prior_sd, log=T) - 
    (pnorm(q=5-Delta_0_new, mean=0, sd=prior_sd, log=T) - pnorm(q=-Delta_0_new, mean=0, sd=prior_sd, log=T)) - 
    (dnorm(x=mu_tar-mu_ntar-Delta_0_old, mean=0, sd=prior_sd, log=T) - 
       (pnorm(q=5-Delta_0_old, mean=0, sd=prior_sd, log=T) - pnorm(q=-Delta_0_old, mean=0, sd=prior_sd, log=T)))
  # log prior ratio
  log_prior_ratio = dnorm(Delta_0_new, mean=0, sd=prior_sd, log=T) - 
    dnorm(Delta_0_old, mean=0, sd=prior_sd, log=T)
  log_alpha = min(0, log_prop_ratio + log_lhd_ratio + log_prior_ratio)
  if (log(runif(1)) < log_alpha) {
    Delta_0_final = Delta_0_new
    accept_final = 1
  } else {
    Delta_0_final = Delta_0_old
    accept_final = 0
  }
  return (list(Delta_0 = Delta_0_final, accept = accept_final))
}


  