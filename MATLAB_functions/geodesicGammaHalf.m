function [Gamma_A_B_half] = geodesicGammaHalf(A, B)
%geodesic_gamma_half Summary of this function goes here
%   Compute the geodesic mean between covariances A and B with t=0.5
Asqrt = sqrtm(A);
Asqrtinv = inv(Asqrt);
Bsqrt = sqrtm(B);
Bsqrtinv = inv(Bsqrt);
Gamma_A_B_half = Asqrt * sqrtm(Asqrtinv * B * Asqrtinv) * Asqrt;
end