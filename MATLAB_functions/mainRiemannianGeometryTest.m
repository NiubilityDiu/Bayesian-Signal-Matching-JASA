%%
clear
clc

%% pre-computing A & B
A = [1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1];
B = [1 0.25 0.25; 0.25 1 0.25; 0.25 0.25 1];

% geodesic gamma operator:
Gamma_A_B = geodesicGammaHalf(A, B);
Gamma_B_A = geodesicGammaHalf(B, A);

disp(Gamma_A_B);
disp(Gamma_B_A);

%% generalize to regular covariance matrices
n=30;
sample_A = mvnrnd(zeros(3, 1), A, n);
sample_B = mvnrnd(zeros(3, 1), B, n);
A_e = cov(sample_A);
B_e = cov(sample_B);

% geodesic gamma operator:
Gamma_A_B_e = geodesicGammaHalf(A_e, B_e);
Gamma_B_A_e = geodesicGammaHalf(B_e, A_e);

disp(Gamma_A_B_e);
disp(Gamma_B_A_e);

%% toeplitz matrix
A_toe = toeplitz([3 2 1]);
B_toe = toeplitz([4 2 1]);
Gamma_A_B_toe = geodesicGammaHalf(A_toe, B_toe);
Gamma_B_A_toe = geodesicGammaHalf(B_toe, A_toe);

disp(Gamma_A_B_toe);
disp(Gamma_B_A_toe);