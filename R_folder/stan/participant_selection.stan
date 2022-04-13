//
// participant-selection with classifier score-based data
//

functions {
  real mu_binary_choose(int y_label, real mu_tar, real mu_ntar) {
    real mu_final;
    if (y_label == 1) {
      mu_final = mu_tar;
    } else {
      mu_final = mu_ntar;
    }
    return mu_final;
  } 
}


data {
  // EEG Signal matrix and y labels
  // We use source / new to distinguish among participants,
  // and tar/ ntar to distinguish between P300 responses.
  
  int <lower=1> sub_n_source;
  int <lower=1> total_size_source;
  int <lower=1> size_new;
  vector[sub_n_source] size_source; // a vector of sample size each source participant.
  vector[total_size_source] x_source;
  int y_source[total_size_source];
  int index_source[total_size_source]; // an index vector of participant assignment.
  vector[size_new] x_new;
  int y_new[size_new];
  
  vector[sub_n_source] mu_source_diff_alpha;
  vector[sub_n_source] mu_source_diff_beta;
  vector[sub_n_source] eta_source_alpha;
  vector[sub_n_source] eta_source_beta;
  real <lower=0> mu_new_diff_alpha;
  real <lower=0> mu_new_diff_beta;
  real <lower=0> eta_new_alpha;
  real <lower=0> eta_new_beta;
}


parameters {
  real<lower=0> mu_source_diff[sub_n_source];
  real mu_source_ntar[sub_n_source];
  vector <lower=0> [sub_n_source] eta_source;
  // real <lower=0> eta_source;  // assume common variance for source participants, model parsimony
  vector <lower=0, upper=1> [sub_n_source] phi_source;
  
  real<lower=0> mu_new_diff;
  real mu_new_ntar; 
  // real <lower=0, upper=1> sigma_new;
  real <lower=0> eta_new;
}


transformed parameters {
  vector <lower=0> [sub_n_source] sigma_source;
  //real <lower=0> sigma_source;
  real <lower=0> sigma_new;
  real mu_source_tar[sub_n_source];
  real mu_new_tar;
  // for (i in 1:sub_n_source) {
  //   sigma_source[i] = sqrt(inv(eta_source[i]));  // avoid using 1/eta in the syntax
  // }
  // sigma_source = sqrt(inv(eta_source));
  sigma_new = sqrt(inv(eta_new));
  for (i in 1:sub_n_source) {
     mu_source_tar[i] = mu_source_ntar[i] + mu_source_diff[i];
     sigma_source[i] = sqrt(inv(eta_source[i]));
  }
  mu_new_tar = mu_new_ntar + mu_new_diff;
}

model {
  // prior
  for (i in 1:sub_n_source) {
    mu_source_ntar[i] ~ normal(0, 1);
    mu_source_diff[i] ~ gamma(mu_source_diff_alpha[i], mu_source_diff_beta[i]);  
    // based on the difference means between target and non-target
    
    eta_source[i] ~ gamma(eta_source_alpha[i], eta_source_beta[i]);
    phi_source[i] ~ beta(0.5, 0.5);
  }

  mu_new_ntar ~ normal(0, 1);
  mu_new_diff ~ gamma(mu_new_diff_alpha, mu_new_diff_beta);
  eta_new ~ gamma(eta_new_alpha, eta_new_beta);

  // likelihood
  // source participant
  for (j in 1:total_size_source) {
    if (y_source[j] == 1) {
      // target stimulus, then decide if it is from new data or source data.
      target += log_mix(phi_source[index_source[j]], 
      normal_lpdf(x_source[j] | mu_new_tar, sigma_new),
      normal_lpdf(x_source[j] | mu_source_tar[index_source[j]], sigma_source[index_source[j]]));
    } else {
      // non-target stimulus, then decide if it is from new data or source data.
      target += log_mix(phi_source[index_source[j]],
      normal_lpdf(x_source[j] | mu_new_ntar, sigma_new),
      normal_lpdf(x_source[j] | mu_source_ntar[index_source[j]], sigma_source[index_source[j]]));
    }
  }
  // target participant
  for (l in 1:size_new) {
    x_new[l] ~ normal(mu_binary_choose(y_new[l], mu_new_tar, mu_new_ntar), sigma_new);
  }
}

