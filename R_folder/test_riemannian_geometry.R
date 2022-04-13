rm(list = ls(all.names=T))
library(mvtnorm)
# Create two covariance matrices
parent_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode/Chapter_3/R_folder'
source(Sys.glob(file.path(parent_dir, 'self_R_fun', 'self_defined_fun.R')))
a = rmvnorm(n=100, mean=rep(0, 3), sigma=matrix(c(1, 0.7, 0, 0.7, 1, 0.7, 0, 0.7, 1), nrow=3, byrow=T))
b = rmvnorm(n=100, mean=rep(0, 3), sigma=matrix(c(1, 0.3, 0, 0.3, 1, 0.3, 0, 0.3, 1), nrow=3, byrow=T))

cov_a = cov(a)
cov_b = cov(b)
