import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import io as sio

parent_path_local = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/' \
                    'Dataset and Rcode/EEG_MATLAB_data'
parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'


class EmpiricalLDA:

    # global constant
    channel_num = 16

    def __init__(self, local_use, sub_name, channel_sub=None, *args, **kwargs):
        self.local_use = local_use
        self.sub_name = sub_name
        self.channel_sub = channel_sub - 1
        self.e_sub_dim = len(self.channel_sub)

        if self.local_use:
            self.parent_dir = parent_path_local
        else:
            self.parent_dir = parent_path_slurm

    def import_train_file(self, trn_file_name):
        mat_dir = '{}/TRN_files/{}/{}_{}.mat'.format(
            self.parent_dir, self.sub_name, self.sub_name, trn_file_name
        )
        mat_obj = sio.loadmat(mat_dir)
        mat_obj_type = mat_obj['Type']
        mat_obj_signal = mat_obj['Signal']
        n, p = mat_obj_signal.shape
        window_len = int(p / self.channel_num)
        if self.channel_sub is not None:
            mat_obj_signal = np.reshape(
                mat_obj_signal, [n, self.channel_num, window_len]
            )
            mat_obj_signal = np.reshape(
                mat_obj_signal[:, self.channel_sub, :], [n, self.e_sub_dim * window_len]
            )
        return mat_obj_signal, mat_obj_type

    def compute_empirical_stats_source(self, trn_file_name):
        mat_obj_signal, mat_obj_type = self.import_train_file(trn_file_name)
        # clf = LDA(solver='lsqr', shrinkage='auto')
        # clf.fit(mat_obj_signal, np.squeeze(mat_obj_type, axis=-1))
        signal_mean_tar = np.mean(mat_obj_signal[mat_obj_type[:,0] == 1, :], axis=0)
        signal_mean_ntar = np.mean(mat_obj_signal[mat_obj_type[:, 0] == -1, :], axis=0)
        signal_cov = np.cov(mat_obj_signal, rowvar=False)
        return signal_mean_tar, signal_mean_ntar, signal_cov

    @staticmethod
    def compute_empirical_stats_new(mat_obj_signal, mat_obj_type):
        signal_mean_tar = np.mean(mat_obj_signal[mat_obj_type[:,0] == 1, :], axis=0)
        signal_mean_ntar = np.mean(mat_obj_signal[mat_obj_type[:, 0] == -1, :], axis=0)
        signal_cov = np.cov(mat_obj_signal, rowvar=False)
        return signal_mean_tar, signal_mean_ntar, signal_cov

    @staticmethod
    def compute_lda_parameter(mean_tar, mean_ntar, cov_both):
        mean_diff = mean_ntar - mean_tar
        mean_sum = mean_tar + mean_ntar
        w_vector = np.linalg.solve(cov_both, mean_diff[:, np.newaxis])
        b_intercept = - 0.5 * np.matmul(np.transpose(w_vector), mean_sum[:, np.newaxis])
        return w_vector, b_intercept


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def compute_bhattacharyya_distance(mu1, mu2, cov1, cov2):
    """
    :param mu1: 2d-array, (window_len, 1)
    :param mu2: 2d-array, (window_len, 1)
    :param cov1: square matrix, (window_len, window_len)
    :param cov2: square matrix, (window_len, window_len)
    :return: A scalar, notice that the function only applies to multivariate normal distribution.
    """
    feature_len = cov1.shape[0]
    ridge_alpha = 0.1 * np.ones([feature_len])
    cov1 = cov1 + np.diag(ridge_alpha)
    cov2 = cov2 + np.diag(ridge_alpha)
    cov = (cov1 + cov2) / 2
    cov_chol_inv = np.linalg.inv(np.linalg.cholesky(cov))
    cov_inv = np.matmul(np.transpose(cov_chol_inv), cov_chol_inv)
    mu_diff = mu1 - mu2
    quad_comp = 1/8 * np.matmul(np.matmul(np.transpose(mu_diff), cov_inv), mu_diff)
    quad_comp = np.squeeze(quad_comp)  # convert from (1, 1) to scalar

    _, logdet_cov = np.linalg.slogdet(cov)
    cov1_chol = np.linalg.cholesky(cov1)
    cov2_chol = np.linalg.cholesky(cov2)
    det_cov1_chol = np.linalg.det(cov1_chol)
    det_cov2_chol = np.linalg.det(cov2_chol)
    cov_comp = 1/2 * (logdet_cov - np.log(det_cov1_chol) - np.log(det_cov2_chol))

    return quad_comp + cov_comp
