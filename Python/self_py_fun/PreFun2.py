import os
import lzma
import seaborn as sns
import json, pickle
import numpy as np
import scipy.stats as stats
from scipy import io as sio
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
from sklearn.decomposition import SparsePCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from self_py_fun.EEGGlobal import *

parent_path_local = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/' \
                    'Dataset and Rcode/EEG_MATLAB_data'
parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'

# Global subject information
# 41 subjects with K protocols
# K_num_ids = [106, 107, 108, 111, 112,
#              113, 114, 115, 117, 118,
#              119, 120, 121, 122, 123,
#              143, 145, 146, 147, 151,
#              152, 154, 155, 156, 158,
#              159, 160, 166, 167, 171,
#              172, 177, 178, 179, 183,
#              184, 185, 190, 191, 212, 223]
#
# M_num_ids = np.array([131, 132, 133, 134, 135,
#                       136, 138, 139, 140, 141,
#                       142, 144, 148, 149])
# sub_name_ls = []
# for K_num_id in K_num_ids:
#     K_sub_name_id = 'K{}'.format(K_num_id)
#     sub_name_ls.append(K_sub_name_id)
#
# for M_num_id in M_num_ids:
#     M_sub_name_id = 'M{}'.format(M_num_id)
#     sub_name_ls.append(M_sub_name_id)
# sub_name_len = len(sub_name_ls)

rcp_screen = np.reshape(np.arange(0, 36), [6, 6]) + 1
stimulus_group_set = [
    rcp_screen[0, :], rcp_screen[1, :], rcp_screen[2, :], rcp_screen[3, :],rcp_screen[4, :], rcp_screen[5, :],
    rcp_screen[:, 0], rcp_screen[:, 1], rcp_screen[:, 2], rcp_screen[:, 3], rcp_screen[:, 4], rcp_screen[:, 5]
]


class EEGPreFun2:
    # Notice that the global constants hold for K-protocol with RCP designs.
    num_electrode = 16
    channel_names = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz',
                     'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz',
                     'P4', 'PO7', 'PO8', 'Oz']
    num_letter = 19  # only for TRN files
    num_rep = 12
    row_column_length = 6  # valid for both K and M protocol files
    char_size = 36

    # def pre_process_raw_signal(
    #         self, local_bool, data_type, sub_name, raw_file_name, band_low, band_upp,
    #         dec_factor, eeg_window, channel_name=None
    # ):
    #     eng1 = matlab.engine.start_matlab()
    #     eeg_down_long_dat = eng1.extractRawEEGRCP(
    #         sub_name, raw_file_name, data_type, band_low, band_upp, dec_factor, local_bool
    #     )
    #     eng1.exit()
    #
    #     eng2 = matlab.engine.start_matlab()
    #     eeg_down_trunc_dat = eng2.truncMatTrainRCP(
    #         eeg_down_long_dat, eeg_window
    #     )
    #     eng2.exit()
    #     # Convert matlab.double to python numpy array
    #     for dict_key_name in eeg_down_trunc_dat.keys():
    #         eeg_down_trunc_dat[dict_key_name] = np.array(eeg_down_trunc_dat[dict_key_name])
    #
    #     if channel_name is not None:
    #         # Reshape the existing dataset
    #         total_flash_num = eeg_down_trunc_dat['Signal'].shape[0]
    #         eeg_down_trunc_dat['Signal'] = np.reshape(
    #             eeg_down_trunc_dat['Signal'], [total_flash_num, self.num_electrode, eeg_window]
    #         )
    #
    #         electrode_name_num = []
    #         for name_id in channel_name:
    #             electrode_name_num.append(self.channel_names.index(name_id))
    #         print('electrode_name_num = {}'.format(electrode_name_num))
    #
    #         eeg_down_trunc_dat['Signal'] = np.reshape(
    #             eeg_down_trunc_dat['Signal'][:, electrode_name_num, :],
    #             [total_flash_num, len(channel_name) * eeg_window]
    #         )
    #     eeg_down_trunc_dat['ID'] = sub_name
    #     return eeg_down_trunc_dat

    # def compute_eeg_swlda_weight(self, eeg_obj, seq_train=None, max_selection=None):
    #
    #     eng3 = matlab.engine.start_matlab()
    #     eeg_signal = eeg_obj['Signal']
    #     eeg_size_total = eeg_signal.shape[0]
    #     feature_length = eeg_signal.shape[1]
    #     eeg_type = eeg_obj['Type']
    #
    #     eeg_signal = matlab.double(eeg_signal.tolist())
    #     eeg_type = matlab.double(eeg_type.tolist())
    #
    #     if seq_train is None:
    #         num_letter = len(eeg_obj['Text'])
    #         seq_train = int(eeg_size_total / num_letter / self.num_rep)
    #         print('seq_train = {}'.format(seq_train))
    #
    #     if max_selection is None:
    #         max_selection = int(0.3 * feature_length)
    #     b, se, pval, in_model, _ = matlab.train_SWLDAmatlab(
    #         eeg_signal, eeg_type, max_selection, nargout=5
    #     )
    #     eng3.exit()
    #
    #     swlda_obj = {
    #         'b': np.array(b),
    #         'in_model': np.array(in_model),
    #         'pval': np.array(pval),
    #         'seq_train': seq_train
    #     }
    #     return swlda_obj

    @staticmethod
    def save_trunc_eeg_obj(
            trunc_eeg_obj, local_bool, data_type, sub_name, raw_file_name
    ):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)

        sio.savemat(
            '{}/{}/{}.mat'.format(parent_path, sub_name, raw_file_name), trunc_eeg_obj
        )
        print('Truncated EEG is finished.')

    def save_eeg_swlda_weights(
            self, swlda_obj,
            local_bool, data_type, sub_name, raw_file_name,
            band_upp, dec_factor, channel_name
    ):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)

        if channel_name is None:
            channel_name = self.channel_names

        # Create swLDA folder if not yet
        swlda_dir = '{}/{}/swLDA'.format(parent_path, sub_name)
        if not os.path.exists(swlda_dir):
            os.mkdir(swlda_dir)

        sio.savemat('{}/{}.mat'.format(swlda_dir, raw_file_name),
                    {
                        'b': swlda_obj['b'],
                        'in_model': swlda_obj['in_model'],
                        'pval': swlda_obj['pval'],
                        'seq_train': swlda_obj['seq_train'],
                        'ID': sub_name,
                        'band_low': 0.5,
                        'band_upp': band_upp,
                        'dec_factor': dec_factor,
                        'channel_name': channel_name
                    })

    @staticmethod
    def import_trunc_eeg_obj(local_bool, data_type, sub_name, raw_file_name):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)
        swlda_obj_dir = '{}/{}/{}.mat'.format(parent_path, sub_name, raw_file_name)
        # print(swlda_obj_dir)
        swlda_obj = sio.loadmat(swlda_obj_dir)
        # swlda_obj_keys, _ = zip(*swlda_obj.items())
        # print(swlda_obj_keys)
        return swlda_obj

    @staticmethod
    def import_eeg_swlda_weights(
            local_bool, data_type, sub_name, raw_file_name, source_letter_num
    ):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)
        if source_letter_num:
            weight_dir = '{}/{}/swLDA/source_letter_num_{}/{}.mat'.format(
                parent_path, sub_name, source_letter_num, raw_file_name
            )
        else:
            weight_dir = '{}/{}/swLDA/{}.mat'.format(parent_path, sub_name, raw_file_name)
        print(weight_dir)
        swlda_weight_obj = sio.loadmat(weight_dir)
        swlda_obj_keys, _ = zip(*swlda_weight_obj.items())
        # print(swlda_obj_keys[3:])
        return swlda_weight_obj

    @staticmethod
    def swlda_predict_classifier_score(
            b, in_model, eeg_signals_trun
    ):
        b_inmodel = np.multiply(np.transpose(in_model), b)
        score = np.matmul(eeg_signals_trun, b_inmodel)
        return score

    @staticmethod
    def swlda_predict_binary_likelihood(pred_score, mu_tar, mu_ntar, std_common):
        score_size = pred_score.shape[0]
        pred_score_binary = np.zeros_like(pred_score) - 1
        for i in range(score_size):
            log_pdf_tar_i = stats.norm.logpdf(pred_score[i, 0], loc=mu_tar, scale=std_common)
            log_pdf_ntar_i = stats.norm.logpdf(pred_score[i, 0], loc=mu_ntar, scale=std_common)
            if log_pdf_tar_i > log_pdf_ntar_i:
                pred_score_binary[i, 0] = 1
        return pred_score_binary

    def ml_predict_letter_naive(
            self, predict_prob, eeg_code, letter_dim, repet_pred, letter_table
    ):
        r"""
        :param predict_prob: 2d-array probability, (feature_vector_dim, 1)
        :param eeg_code: ultimately converted to 1d-array
        :param letter_dim: integer
        :param repet_pred: integer
        :param letter_table: list
        :return:
        """
        assert predict_prob.shape == (letter_dim * repet_pred * self.num_rep, 1), \
            print('Inconsistent dimension of predict_prob.')

        eeg_code = np.reshape(eeg_code, [letter_dim * repet_pred * self.num_rep])
        single_score_row = np.zeros([int(self.num_rep / 2), letter_dim * repet_pred])
        single_score_col = np.zeros([int(self.num_rep / 2), letter_dim * repet_pred])

        for i in range(int(self.num_rep / 2)):
            single_score_row[i, :] = predict_prob[np.where(eeg_code == i + 1)[0], 0]
            single_score_col[i, :] = predict_prob[np.where(eeg_code == i + self.row_column_length + 1)[0], 0]

        single_score_row = np.reshape(
            single_score_row, [int(self.num_rep / 2), letter_dim, repet_pred]
        )
        single_score_col = np.reshape(
            single_score_col, [int(self.num_rep / 2), letter_dim, repet_pred]
        )

        # print('single_score_row has shape {}'.format(single_score_row.shape))
        # Compute the prediction based on single seq (row + col)
        arg_max_single_row = np.argmax(single_score_row, axis=0) + 1
        arg_max_single_col = np.argmax(single_score_col, axis=0) + self.row_column_length + 1

        # cumulative
        cum_score_row = np.cumsum(single_score_row, axis=-1)
        cum_score_col = np.cumsum(single_score_col, axis=-1)
        arg_max_cum_row = np.argmax(cum_score_row, axis=0) + 1
        arg_max_cum_col = np.argmax(cum_score_col, axis=0) + self.row_column_length + 1

        letter_single_mat = []
        letter_cum_mat = []

        for i in range(letter_dim):
            for j in range(repet_pred):
                letter_single_mat.append(self.determine_letter(
                    arg_max_single_row[i, j], arg_max_single_col[i, j], letter_table)
                )
                letter_cum_mat.append(self.determine_letter(
                    arg_max_cum_row[i, j], arg_max_cum_col[i, j], letter_table)
                )

        letter_single_mat = np.reshape(np.array(letter_single_mat), [letter_dim, repet_pred])
        letter_cum_mat = np.reshape(np.array(letter_cum_mat), [letter_dim, repet_pred])
        ml_pred_dict = {
            "single": letter_single_mat.astype('<U5'),
            "cum": letter_cum_mat.astype('<U5')
        }
        return ml_pred_dict

    def _ml_predict_letter_likelihood_unit(
            self, char_prob, stimulus_score, stimulus_code, mu_tar, mu_ntar, std_common,
            unit_stimulus_set
    ):
        """
        Apply the bayesian naive dynamic stopping criterion
        :param char_prob:
        :param stimulus_score:
        :param stimulus_code:
        :param mu_tar:
        :param mu_ntar:
        :param std_common:
        :param unit_stimulus_set:
        :return:
        """
        char_prob_post = np.copy(np.log(char_prob))
        for s_id in range(self.num_rep):
            for char_id in range(1, self.char_size + 1):
                if char_id in unit_stimulus_set[stimulus_code[s_id]-1]:
                    char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                             stats.norm.logpdf(stimulus_score[s_id], loc=mu_tar, scale=std_common)
                else:
                    char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                             stats.norm.logpdf(stimulus_score[s_id], loc=mu_ntar, scale=std_common)
        char_prob_post = char_prob_post - np.max(char_prob_post)
        char_prob_post = np.exp(char_prob_post)
        char_prob_post = char_prob_post / np.sum(char_prob_post)
        return char_prob_post

    def ml_predict_letter_likelihood(
            self, stimulus_score, stimulus_code, letter_dim, repet_pred,
            mu_tar, mu_ntar, std_common, unit_stimulus_set, letter_table_ls
    ):
        stimulus_score = np.reshape(stimulus_score, [letter_dim, repet_pred, self.num_rep])
        stimulus_code = np.reshape(stimulus_code, [letter_dim, repet_pred, self.num_rep])
        char_prob_mat = np.zeros([letter_dim, repet_pred+1, self.char_size]) + 1 / self.char_size
        for letter_id in range(letter_dim):
            for seq_id in range(repet_pred):
                char_prob_mat[letter_id, seq_id+1, :] = self._ml_predict_letter_likelihood_unit(
                    char_prob_mat[letter_id, seq_id, :], stimulus_score[letter_id, seq_id, :],
                    stimulus_code[letter_id, seq_id, :], mu_tar, mu_ntar, std_common, unit_stimulus_set
                )
        char_prob_mat = char_prob_mat[:, 1:, :]
        argmax_prob_mat = np.argmax(char_prob_mat, axis=-1)
        char_max_mat = np.zeros_like(argmax_prob_mat).astype('<U5')
        for letter_id in range(letter_dim):
            for seq_id in range(repet_pred):
                char_max_mat[letter_id, seq_id] = letter_table_ls[argmax_prob_mat[letter_id, seq_id]]
        return char_max_mat, char_prob_mat

    def determine_letter(self, row_index, column_index, letter_table):
        r"""
        :param letter_table:
        :param row_index: 1-6
        :param column_index: 7-12
        :return:
        """
        assert 1 <= row_index <= self.row_column_length and self.row_column_length + 1 <= column_index <= self.num_rep
        letter_index = (row_index - 1) * self.row_column_length + (column_index - self.row_column_length)
        return letter_table[letter_index - 1]

    @staticmethod
    def save_eeg_ml_predict_naive(
            ml_pred_obj,
            local_bool, sub_name, method_name, raw_file_name
    ):
        data_type = 'FRT_files'
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)

        # Create swLDA folder if not yet
        ml_dir = '{}/{}/{}'.format(parent_path, sub_name, method_name)
        if not os.path.exists(ml_dir):
            os.mkdir(ml_dir)

        sio.savemat('{}/{}.mat'.
                    format(ml_dir, raw_file_name),
                    {
                        'single': ml_pred_obj['single'],
                        'cum': ml_pred_obj['cum']
                    })

    @staticmethod
    def save_eeg_ml_predict_likelihood(
            ml_pred_obj, local_bool, sub_name, method_name, raw_file_name
    ):
        data_type = 'FRT_files'
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)

        # Create swLDA folder if not yet
        ml_dir = '{}/{}/{}'.format(parent_path, sub_name, method_name)
        if not os.path.exists(ml_dir):
            os.mkdir(ml_dir)

        sio.savemat('{}/{}.mat'.
                    format(ml_dir, raw_file_name),
                    {
                        # 'single': ml_pred_obj['single'],
                        'cum': ml_pred_obj
                    })

    @staticmethod
    def import_cluster_group(local_bool, data_type, raw_file_name):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)
        cluster_assign_dir = '{}/{}.mat'.format(parent_path, raw_file_name)
        cluster_assign = sio.loadmat(cluster_assign_dir)
        cluster_assign_keys, _ = zip(*cluster_assign.items())
        # print(cluster_assign_keys)
        return cluster_assign

    def summarize_accuracy_naive(
            self, local_bool, pred_binary_arr, pred_prob_arr, true_y_label, eeg_code,
            sub_name, true_letter, seq_test, method_name, letter_table_ls
    ):
        # binary classification
        pred_binary_accuracy = np.mean(pred_binary_arr == true_y_label)
        auc_val = roc_auc_score(true_y_label, pred_prob_arr)

        # character prediction
        pred_prob_letter = self.ml_predict_letter_naive(
            pred_prob_arr, eeg_code, len(true_letter), seq_test, letter_table_ls
        )
        letter_accuracy = compute_prediction_accuracy_inside(
            true_letter, pred_prob_letter['cum'].astype('<U5')
        )
        result_dict = {
            'num_letter': len(true_letter),
            'seq_test': seq_test,
            'binary': pred_binary_accuracy,
            'auc': auc_val,
            'letter': letter_accuracy['Accuracy'] / letter_accuracy['Total']
        }

        if local_bool:
            parent_path = '{}/FRT_files'.format(parent_path_local)
        else:
            parent_path = '{}/FRT_files'.format(parent_path_slurm)
        if not os.path.exists('{}/{}/{}'.format(parent_path, sub_name, method_name)):
            os.mkdir('{}/{}/{}'.format(parent_path, sub_name, method_name))

        # Save result_dict
        # confusion_mat = confusion_matrix(true_y_label, pred_binary_arr)
        # names = ['Non-target', 'Target']
        # sio.savemat(
        #     '{}/{}/{}/Accuracy_dict_{}.mat'.format(
        #         parent_path, sub_name, method_name, frt_file_name),
        #     result_dict
        # )
        # with open('{}/{}/{}/Accuracy_dict_{}.json'.format(
        #         parent_path, sub_name, method_name, frt_file_name), 'w') as file:
        #     json.dump(result_dict, file)
        # file.close()
        # # Save confusion matrix
        # confusion_mat_pdf = bpdf.PdfPages('{}/{}/{}/Confusion_matrix_{}.pdf'.format(
        #     parent_path, sub_name, method_name, frt_file_name
        # ))
        # plt.figure(figsize=(12, 12))
        # ConfusionMatrixDisplay(confusion_mat, display_labels=names).plot()
        # confusion_mat_pdf.savefig()
        # confusion_mat_pdf.close()
        return result_dict

    def summarize_accuracy_likelihood(
            self, local_bool, pred_binary_arr, pred_prob_arr, true_y_label, eeg_code,
            sub_name, true_letter, seq_test, frt_file_name, method_name,
            mu_tar, mu_ntar, std_common, unit_stimulus_set, letter_table_ls
    ):
        # binary classification
        pred_binary_accuracy = np.mean(pred_binary_arr == true_y_label)
        tn, fp, fn, tp = confusion_matrix(true_y_label, pred_binary_arr).ravel()
        auc_val = roc_auc_score(true_y_label, pred_prob_arr)

        pred_prob_letter, pred_prob_mat = self.ml_predict_letter_likelihood(
            pred_prob_arr, eeg_code, len(true_letter), seq_test, mu_tar, mu_ntar, std_common,
            unit_stimulus_set, letter_table_ls
        )
        # self.save_eeg_ml_predict_likelihood(
        #     pred_prob_letter, local_bool, sub_name, method_name,
        #     'Prediction_{}'.format(frt_file_name)
        # )
        letter_accuracy = compute_prediction_accuracy_inside(
            true_letter, pred_prob_letter
        )

        result_dict = {
            'ID': sub_name,
            'num_letter': len(true_letter),
            'seq_test': seq_test,
            'binary': pred_binary_accuracy,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'auc': auc_val,
            'letter': letter_accuracy['Accuracy'] / letter_accuracy['Total']
        }

        if local_bool:
            parent_path = '{}/FRT_files'.format(parent_path_local)
        else:
            parent_path = '{}/FRT_files'.format(parent_path_slurm)
        if not os.path.exists('{}/{}/{}'.format(parent_path, sub_name, method_name)):
            os.mkdir('{}/{}/{}'.format(parent_path, sub_name, method_name))
        return result_dict


class EEGsPCA(SparsePCA):
    def __init__(self, n_comp_max, alpha=1, ridge_alpha=0.1, *args, **kwargs):
        super(SparsePCA, self).__init__(*args, **kwargs)
        self.n_com_max = n_comp_max
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        # For feature full, we need to apply sparse PCA.
        self.spca_x = SparsePCA(
            n_components=self.n_com_max, random_state=0, alpha=self.alpha, ridge_alpha=self.ridge_alpha
        )

    def fit_spca_obj(self, x_data):
        self.spca_x.fit(x_data)
        spca_x_comps = self.spca_x.components_
        spca_x_mean = self.spca_x.mean_
        x_data_spca = self.spca_x.fit_transform(x_data)
        _, x_data_spca_r = np.linalg.qr(x_data_spca)
        spca_x_var = np.cumsum(np.diag(x_data_spca_r ** 2))
        spca_x_var = spca_x_var / spca_x_var[self.n_com_max - 1]
        return spca_x_comps, spca_x_mean, x_data_spca, spca_x_var

    def plot_spca_obj_var(self, spca_x_var, local_bool, data_type):
        if local_bool:
            parent_path = '{}/{}'.format(parent_path_local, data_type)
        else:
            parent_path = '{}/{}'.format(parent_path_slurm, data_type)
        plot_1_pdf = bpdf.PdfPages('{}/feature_full_pca_var.pdf'.format(parent_path))
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, self.n_com_max + 1), spca_x_var, marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        # plt.show()
        plot_1_pdf.savefig()
        plot_1_pdf.close()

    @staticmethod
    def recover_from_spca_comp(spca_comps, spca_mean, new_data_spca):
        x_recover = new_data_spca.dot(spca_comps) + spca_mean - spca_mean
        return x_recover


class EEGClusterFit(EEGPreFun2):
    # Global constants
    key_list = ['Signal', 'Code', 'Type', 'Text']
    # examine whether reducing the training sample size will affect the clustering prediction accuracy
    tar_char_size = 5
    ntar_char_size = 5

    def __init__(self, cluster_assign_arr, raw_file_name_trunc, cluster_size, **kwargs):
        super(EEGClusterFit, self).__init__()
        self.cluster_assign_arr = cluster_assign_arr
        self.raw_file_name_trunc = raw_file_name_trunc
        self.cluster_size = cluster_size
        if 'tar_char_size' in kwargs.keys():
            self.tar_char_size = kwargs['tar_char_size']
        if 'ntar_char_size' in kwargs.keys():
            self.ntar_char_size = kwargs['ntar_char_size']

    def create_data_obj_exclude(
            self, local_use, candidate_ids, data_type, raw_file_name
    ):
        # This function is used when there is no conflict
        # between candidate_ids and target_id.
        target_id_data_clust_obj = {'Signal': [], 'Code': [], 'Type': [], 'Text': []}
        for sub_name_id in candidate_ids:
            raw_file_name_2 = '{}_{}'.format(sub_name_id, raw_file_name)
            sub_name_id_obj = self.import_trunc_eeg_obj(local_use, data_type, sub_name_id, raw_file_name_2)
            text_length = sub_name_id_obj['Text'].shape[0]
            total_stimulus_size = sub_name_id_obj['Signal'].shape[0]
            unit_stimulus_size = int(total_stimulus_size / text_length / 12)
            for key_id in self.key_list:
                if key_id == 'Text':
                    target_id_data_clust_obj[key_id].extend(sub_name_id_obj[key_id][:self.ntar_char_size])
                else:
                    target_id_data_clust_obj[key_id].append(
                        sub_name_id_obj[key_id][:unit_stimulus_size * self.ntar_char_size, :]
                    )
        for key_id in self.key_list:
            target_id_data_clust_obj[key_id] = np.vstack(target_id_data_clust_obj[key_id])
        return target_id_data_clust_obj

    def fit_swlda_obj_out_of_cluster(self, local_use, data_type, max_selection=None):
        swlda_obj_out_clust_dict = {}

        for clust_id in range(self.cluster_size):
            clust_group_id = self.cluster_assign_arr[self.cluster_assign_arr[:, 0] == str(clust_id), 1]
            print('Cluster id: {} \n Cluster labels: {} \n'.format(clust_id, clust_group_id))

            trn_data_clust_id = self.create_data_obj_exclude(
                local_use, clust_group_id, data_type, self.raw_file_name_trunc
            )
            # Notice that seq_train_clust_id may not be an integer due to K154 and K190.
            seq_train_clust_id = trn_data_clust_id['Signal'].shape[0] / len(trn_data_clust_id['Text']) / 12

            swlda_obj_clust_id = self.compute_eeg_swlda_weight(
                trn_data_clust_id, seq_train_clust_id, max_selection
            )
            weight_name_clust_id = 'Weight_clust_{}_id_{}_out'.format(self.cluster_size, clust_id)
            swlda_obj_out_clust_dict[weight_name_clust_id] = swlda_obj_clust_id
        return swlda_obj_out_clust_dict

    def create_data_obj_include(
            self, local_bool, candidate_ids, target_id, data_type, raw_file_name
    ):
        key_lists = ['Signal', 'Code', 'Type', 'Text']
        target_id_data_clust_obj = {}
        for key_list in key_lists:
            target_id_data_clust_obj[key_list] = []
        for sub_name_id in candidate_ids:
            if sub_name_id == target_id:
                pass
            else:
                raw_file_name_2 = '{}_{}'.format(sub_name_id, raw_file_name)
                sub_name_id_obj = self.import_trunc_eeg_obj(local_bool, data_type, sub_name_id, raw_file_name_2)
                total_flash_size = sub_name_id_obj['Signal'].shape[0]
                unit_flash_size = int(total_flash_size / len(sub_name_id_obj['Text']))

                for key_id in key_lists:
                    if key_id == 'Text':
                        target_id_data_clust_obj[key_id].extend(sub_name_id_obj[key_id][:self.ntar_char_size])
                    else:
                        target_id_data_clust_obj[key_id].append(
                            sub_name_id_obj[key_id][:unit_flash_size * self.ntar_char_size, :]
                        )

        # We add a portion of data from target subject for both cluster prediction.
        raw_file_name_target = '{}_{}'.format(target_id, raw_file_name)
        target_id_obj = self.import_trunc_eeg_obj(local_bool, data_type, target_id, raw_file_name_target)
        total_flash_size_target = target_id_obj['Signal'].shape[0]
        unit_flash_size = int(total_flash_size_target / target_id_obj['Text'].shape[0])
        for key_id in key_lists:
            if key_id == 'Text':
                target_id_data_clust_obj[key_id].extend(target_id_obj[key_id][:self.tar_char_size])
            else:
                target_id_data_clust_obj[key_id].append(
                    target_id_obj[key_id][:unit_flash_size * self.tar_char_size, :]
                )
        for key_id in key_lists:
            target_id_data_clust_obj[key_id] = np.vstack(target_id_data_clust_obj[key_id])
        return target_id_data_clust_obj

    def fit_swlda_obj_cluster_exclude(
            self, local_use, data_type, iid, swlda_obj_out_clust_dict
    ):
        target_id = self.cluster_assign_arr[iid, 1]
        swlda_fit_clusts_dict = {}
        for clust_id in range(self.cluster_size):
            final_clust_group = self.cluster_assign_arr[self.cluster_assign_arr[:, 0] == str(clust_id), 1]
            final_clust_group_len = len(final_clust_group)
            print('group {} contains {}'.format(clust_id, final_clust_group))
            if clust_id == int(self.cluster_assign_arr[iid, 0]) and final_clust_group_len > 1:
                # Within-cluster training
                print('Within cluster swLDA fit.')
                print('We exclude all self-TRN file results of {} for fitting.'.format(target_id))
                final_clust_group = np.setdiff1d(final_clust_group, target_id)
                print('final group after exclusion = {}'.format(final_clust_group))
                swlda_target_id_clust_obj = self.create_data_obj_exclude(
                    local_use, final_clust_group.tolist(), data_type, self.raw_file_name_trunc
                )
                total_stimulus_size = swlda_target_id_clust_obj['Signal'].shape[0]
                total_text_length = len(swlda_target_id_clust_obj['Text'])
                seq_train_target_id_clust = total_stimulus_size / total_text_length / 12
                swlda_target_id_clust_wts = self.compute_eeg_swlda_weight(
                    swlda_target_id_clust_obj, seq_train_target_id_clust, None
                )
                weight_clust_name = 'Weight_clust_{}_id_{}_001_BCI_TRN'.format(self.cluster_size, clust_id)
                swlda_fit_clusts_dict[weight_clust_name] = swlda_target_id_clust_wts
            else:
                if clust_id == int(self.cluster_assign_arr[iid, 0]) and final_clust_group_len == 1:
                    print('Within cluster swLDA fit with single cluster.')
                    print('We use self-TRN file results and directly borrow the previously-calculated result.')
                else:
                    print('Out-of cluster swLDA fit.')
                    print('We directly borrow the previously-calculated result.')
                weight_name_clust_id= 'Weight_clust_{}_id_{}'.format(self.cluster_size, clust_id)
                weight_name_clust_1 = '{}_001_BCI_TRN'.format(weight_name_clust_id)
                weight_name_clust_2 = '{}_out'.format(weight_name_clust_id)
                swlda_fit_clusts_dict[weight_name_clust_1] = swlda_obj_out_clust_dict[weight_name_clust_2]
        print('swLDA fit object of target id {} is completed.\n'.format(target_id))
        return swlda_fit_clusts_dict

    def fit_swlda_obj_cluster_include(
            self, local_use, data_type, iid, max_selection
    ):
        target_id = self.cluster_assign_arr[iid, 1]
        swlda_fit_clusts_dict = {}
        for clust_id in range(self.cluster_size):
            if clust_id == int(self.cluster_assign_arr[iid, 0]):
                print('Within cluster swLDA fit.')
            else:
                print('Out-of cluster swLDA fit.')
            print('We use at most {} char(s) of training data from {}.'.format(self.tar_char_size, target_id))
            final_clust_group = self.cluster_assign_arr[self.cluster_assign_arr[:, 0] == str(clust_id), 1]

            target_data_id_clust = self.create_data_obj_include(
                local_use, final_clust_group.tolist(), target_id, data_type, self.raw_file_name_trunc
            )

            total_stimulus_size = target_data_id_clust['Signal'].shape[0]
            total_text_num = target_data_id_clust['Text'].shape[0]
            seq_train_target_id_clust = total_stimulus_size / total_text_num / 12  # may ba a fraction
            swlda_target_id_clust_obj = self.compute_eeg_swlda_weight(
                target_data_id_clust, seq_train_target_id_clust, max_selection
            )
            weight_clust_name = 'Weight_clust_{}_id_{}_001_BCI_TRN'.format(self.cluster_size, clust_id)
            swlda_fit_clusts_dict[weight_clust_name] = swlda_target_id_clust_obj
        return swlda_fit_clusts_dict

    def predict_swlda_obj_iid_cluster(
            self, local_use, data_type, iid, swlda_target_id_fit_obj_clust
    ):
        swlda_predict_clusts_dict = {}
        target_id = self.cluster_assign_arr[iid, 1]
        for cluster_id in range(self.cluster_size):
            print(cluster_id)
            weight_obj_name_clust_id = 'Weight_clust_{}_id_{}_001_BCI_TRN'.format(self.cluster_size, cluster_id)
            swlda_weight_obj_clust_id = swlda_target_id_fit_obj_clust[weight_obj_name_clust_id]
            b_clust_id = swlda_weight_obj_clust_id['b']
            # if not b_clust_id.shape == (400, 1):
            #     b_clust_id = b_clust_id[0, 0]
            in_model_clust_id = swlda_weight_obj_clust_id['in_model']
            # if not in_model_clust_id.shape == (1, 400):
            #     in_model_clust_id = in_model_clust_id[0, 0]
            swlda_predict_clusts_dict[weight_obj_name_clust_id] = {}

            if target_id in FRT_file_name_dict:
                frt_file_name_ls = FRT_file_name_dict[target_id]
            else:
                frt_file_name_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']

            # import the FRT datasets
            for predict_file_name in frt_file_name_ls:
                swlda_predict_clusts_dict[weight_obj_name_clust_id][predict_file_name] = {}
                frt_file_name_trunc = '{}_{}_Truncated_Data'.format(target_id, predict_file_name)
                eeg_FRT_dat = self.import_trunc_eeg_obj(local_use, data_type, target_id, frt_file_name_trunc)
                num_letter = len(eeg_FRT_dat['Text'])
                seq_test = int(eeg_FRT_dat['Signal'].shape[0] / num_letter / self.num_rep)
                print('num_letter = {}, seq_test = {}'.format(num_letter, seq_test))
                letter_table = eeg_FRT_dat['LetterTable']

                swlda_prob_flash = self.swlda_predict_classifier_score(b_clust_id, in_model_clust_id, eeg_FRT_dat['Signal'])
                swlda_prob_letter = self.ml_predict_letter_naive(swlda_prob_flash, eeg_FRT_dat['Code'], num_letter,
                                                                 seq_test, letter_table)
                # print(swlda_prob_letter)
                swlda_prob_letter['single'] = swlda_prob_letter['single'].tolist()
                swlda_prob_letter['cum'] = swlda_prob_letter['cum'].tolist()
                # predict_file_name_2 = 'Predict_clust_{}_id_{}'.format(self.cluster_size, cluster_id)
                swlda_predict_clusts_dict[weight_obj_name_clust_id][predict_file_name] = swlda_prob_letter
        # print(swlda_predict_clusts_dict)
        print('swLDA test object of target id {} is completed.\n'.format(target_id))
        return swlda_predict_clusts_dict

    def import_swlda_obj_iid_cluster(
            self, local_use, iid, **kwargs
    ):
        target_id = self.cluster_assign_arr[iid, 1]
        if local_use:
            parent_path_fit = '{}/{}'.format(parent_path_local, 'TRN_files')
        else:
            parent_path_fit = '{}/{}'.format(parent_path_slurm, 'TRN_files')
        swlda_fit_dir = '{}/{}/swLDA'.format(parent_path_fit, target_id)

        if 'null_id' in kwargs.keys():
            null_id = kwargs['null_id']
            swlda_fit_dir_2 = '{}/null_{}/null_{}_dataset_{}/null_fit.mat'.format(
                swlda_fit_dir, self.cluster_size, self.cluster_size, null_id
            )
            swlda_fit_obj = sio.loadmat(swlda_fit_dir_2)
        elif 'method' in kwargs.keys():
            clust_method = kwargs['method']
            swlda_fit_dir_2 = '{}/{}_{}_label/{}_fit.mat'.format(
                swlda_fit_dir, clust_method, self.cluster_size, clust_method
            )
            swlda_fit_obj = sio.loadmat(swlda_fit_dir_2)
        else:
            swlda_fit_obj = {}

        return swlda_fit_obj

    def save_swlda_obj_iid_cluster(
            self, local_use, iid, swlda_target_id_fit_obj, swlda_target_id_predict_obj,
            **kwargs
    ):
        target_id = self.cluster_assign_arr[iid, 1]
        if local_use:
            parent_path_fit = '{}/{}'.format(parent_path_local, 'TRN_files')
            parent_path_predict = '{}/{}'.format(parent_path_local, 'FRT_files')

        else:
            parent_path_fit = '{}/{}'.format(parent_path_slurm, 'TRN_files')
            parent_path_predict = '{}/{}'.format(parent_path_slurm, 'FRT_files')

        # Create swLDA folder if not yet
        swlda_fit_dir = '{}/{}/swLDA'.format(parent_path_fit, target_id)
        swlda_predict_dir = '{}/{}/swLDA'.format(parent_path_predict, target_id)
        if not os.path.exists(swlda_fit_dir):
            os.mkdir(swlda_fit_dir)
        if not os.path.exists(swlda_predict_dir):
            os.mkdir(swlda_predict_dir)

        # https://realpython.com/python-kwargs-and-args/ for details of args and kwargs
        if 'null_id' in kwargs.keys():
            null_id = kwargs['null_id']

            swlda_fit_dir_2 = '{}/null_{}'.format(swlda_fit_dir, self.cluster_size)
            swlda_predict_dir_2 = '{}/null_{}'.format(swlda_predict_dir, self.cluster_size)
            if not os.path.exists(swlda_fit_dir_2):
                os.mkdir(swlda_fit_dir_2)
            if not os.path.exists(swlda_predict_dir_2):
                os.mkdir(swlda_predict_dir_2)

            swlda_fit_dir_3 = '{}/null_{}_dataset_{}'.format(swlda_fit_dir_2, self.cluster_size, null_id)
            swlda_predict_dir_3 = '{}/null_{}_dataset_{}'.format(swlda_predict_dir_2, self.cluster_size, null_id)
            if not os.path.exists(swlda_fit_dir_3):
                os.mkdir(swlda_fit_dir_3)
            if not os.path.exists(swlda_predict_dir_3):
                os.mkdir(swlda_predict_dir_3)

            sio.savemat('{}/null_fit.mat'.format(swlda_fit_dir_3), swlda_target_id_fit_obj)
            # with open('{}/null_fit.json'.format(swlda_fit_dir_3), 'w') as file:
            #     json.dump(swlda_target_id_fit_obj, file)
            # For multi-layered nested dict, use json
            with open('{}/null_predict.json'.format(swlda_predict_dir_3), 'w') as file:
                json.dump(swlda_target_id_predict_obj, file)
            file.close()

        elif 'method' in kwargs.keys():
            clust_method = kwargs['method']
            swlda_fit_dir_2 = '{}/{}_{}_label'.format(swlda_fit_dir, clust_method, self.cluster_size)
            swlda_predict_dir_2 = '{}/{}_{}_label'.format(swlda_predict_dir, clust_method, self.cluster_size)
            if not os.path.exists(swlda_fit_dir_2):
                os.mkdir(swlda_fit_dir_2)
            if not os.path.exists(swlda_predict_dir_2):
                os.mkdir(swlda_predict_dir_2)

            sio.savemat('{}/{}_fit.mat'.format(swlda_fit_dir_2, clust_method), swlda_target_id_fit_obj)
            with open('{}/{}_predict.json'.format(swlda_predict_dir_2, clust_method), 'w') as file:
                json.dump(swlda_target_id_predict_obj, file)
            file.close()


# Independent self-defined functions below
def fit_predict_cluster(
        local_use, cluster_assign_arr, raw_file_name,
        cluster_size, max_selection, method_type, **kwargs
):
    ClustObj = EEGClusterFit(
        cluster_assign_arr=cluster_assign_arr,
        raw_file_name_trunc=raw_file_name,
        cluster_size=cluster_size, **kwargs
    )

    data_type_fit = 'TRN_files'
    data_type_pred = 'FRT_files'
    # out-of-cluster fit to save running time
    # only used when method_type == 'exclude'.
    # when we perform single cluster analysis, comment out this function as well.
    if cluster_size > 1 and 'exclude' in method_type:
        swlda_obj_out_clust_dict = ClustObj.fit_swlda_obj_out_of_cluster(
            local_use, data_type_fit, max_selection
        )
    else:
        swlda_obj_out_clust_dict = {}

    for iid, target_id in enumerate(cluster_assign_arr[:, 1].tolist()):
        print('Index: {}, Subject Name: {}. \n'.format(iid, target_id))
        if 'include' in method_type:
            # Fit with inclusion criterion
            swlda_target_id_fit_obj_clust = ClustObj.fit_swlda_obj_cluster_include(
                local_use, data_type_fit, iid, max_selection
            )
        else:
            # Fit with exclusion criterion
            swlda_target_id_fit_obj_clust = ClustObj.fit_swlda_obj_cluster_exclude(
                local_use, data_type_fit, iid, swlda_obj_out_clust_dict
            )

        '''
        # import Fit object
        swlda_target_id_fit_obj_clust = ClustObj.import_swlda_obj_iid_cluster(
            local_use, iid, **kwargs
        )
        '''

        # Predict
        swlda_target_id_predict_obj_clust = ClustObj.predict_swlda_obj_iid_cluster(
            local_use, data_type_pred, iid, swlda_target_id_fit_obj_clust
        )
        # Save
        ClustObj.save_swlda_obj_iid_cluster(
            local_use, iid,
            swlda_target_id_fit_obj_clust, swlda_target_id_predict_obj_clust,
            **kwargs
        )


def import_swlda_obj_iid_cluster(
        local_use, target_id, type_name, type_name_id, raw_file_name
):
    if local_use:
        parent_path_predict = '{}/{}'.format(parent_path_local, 'FRT_files')
    else:
        parent_path_predict = '{}/{}'.format(parent_path_slurm, 'FRT_files')

    swlda_predict_dir = '{}/{}/swLDA/{}/{}_dataset_{}/{}.json'.format(
        parent_path_predict, target_id, type_name, type_name, type_name_id, raw_file_name
    )
    # open json file
    swlda_predict_f = open(swlda_predict_dir)
    swlda_predict_obj = json.load(swlda_predict_f)
    swlda_predict_f.close()
    return swlda_predict_obj


def convert_swlda_predict_obj(swlda_predict_obj):
    # convert the nested dict with list to nested dict with numpy array
    first_l_keys = swlda_predict_obj.keys()
    for first_l_key in first_l_keys:
        swlda_predict_obj_l_1 = swlda_predict_obj[first_l_key]
        second_l_keys = swlda_predict_obj_l_1.keys()
        for second_l_key in second_l_keys:
            swlda_predict_obj_l_2 = swlda_predict_obj_l_1[second_l_key]
            third_l_keys = swlda_predict_obj_l_2.keys()
            for third_l_key in third_l_keys:
                swlda_predict_obj_l_3 = swlda_predict_obj_l_2[third_l_key]
                swlda_predict_obj[first_l_key][second_l_key][third_l_key] = np.vstack(swlda_predict_obj_l_3)
    return swlda_predict_obj


def compute_prediction_accuracy(local_use, data_type, target_id, swlda_predict_obj):
    EEGObj = EEGPreFun2()
    true_char_dict = {}
    predict_cum_dict = {}

    # import true character info
    if target_id in FRT_file_name_dict.keys():
        FRT_file_name_ls = FRT_file_name_dict[target_id]
    else:
        FRT_file_name_ls = ['001_BCI_FRT', '002_BCI_FRT', '003_BCI_FRT']
    for frt_file_name in FRT_file_name_ls:
        raw_predict_name = '{}_{}_Truncated_Data'.format(target_id, frt_file_name)
        frt_data_obj = EEGObj.import_trunc_eeg_obj(local_use, data_type, target_id, raw_predict_name)
        true_char_dict[frt_file_name] = frt_data_obj['Text']

    clust_keys = swlda_predict_obj.keys()
    for clust_key in clust_keys:
        predict_cum_dict[clust_key] = {}
        swlda_predict_clust_obj = swlda_predict_obj[clust_key]
        frt_keys = swlda_predict_clust_obj.keys()
        for frt_key in frt_keys:
            # we only focus on cumulative prediction
            swlda_predict_clust_frt_obj = swlda_predict_clust_obj[frt_key]['cum'].astype('<U5')
            predict_cum_dict[clust_key][frt_key] = {}
            true_char_frt_key = true_char_dict[frt_key].astype('<U5')  # fair comparison, upper bounded by 'SPACE'
            # total_char_num = len(true_char_frt_key)
            # predict_cum_dict[clust_key][frt_key]['Total'] = total_char_num
            # total_accuracy = 0
            # for i in range(total_char_num):
            #     accuracy_i = int(swlda_predict_clust_frt_obj[i, -1] in true_char_frt_key[i])
            #     if 'BS' in swlda_predict_clust_frt_obj[i, -1] and 'BS' in true_char_frt_key[i]:
            #         accuracy_i = 1
            #     elif swlda_predict_clust_frt_obj[i, -1] == 'SPACE' and true_char_frt_key[i] == '    ':
            #         accuracy_i = 1
            #     total_accuracy = total_accuracy + accuracy_i
            # predict_cum_dict[clust_key][frt_key]['Accuracy'] = total_accuracy
            predict_cum_dict[clust_key][frt_key] = compute_prediction_accuracy_inside(
                true_char_frt_key, swlda_predict_clust_frt_obj
            )
    return true_char_dict, predict_cum_dict


def compute_prediction_accuracy_inside(
        true_char_arr, predict_char_arr_cum
):
    """
    :param true_char_arr:
    :param predict_char_arr_cum:
    :return:
    We only look at prediction results from the last sequence.
    """
    predict_cum_dict = {}
    total_char_cum = len(true_char_arr)
    predict_cum_dict['Total'] = int(total_char_cum)
    total_accuracy = 0
    for char_id in range(total_char_cum):
        # Remove extra space in the string for comparison
        predict_char_id = predict_char_arr_cum[char_id, -1].strip()
        true_char_id = true_char_arr[char_id].strip()
        accuracy_id = int(predict_char_id in true_char_id)
        if 'BS' in predict_char_id and 'BS' in true_char_id:
            accuracy_id = 1
        elif predict_char_id == 'SPACE' and true_char_id == '':
            accuracy_id = 1
        # print(accuracy_id)
        total_accuracy = total_accuracy + accuracy_id
    predict_cum_dict['Accuracy'] = total_accuracy

    return predict_cum_dict


def combine_prediction_accuracy(predict_cum_dict):
    predict_cum_sum_dict = {}
    clust_keys = predict_cum_dict.keys()
    for clust_key in clust_keys:
        predict_cum_sum_dict[clust_key] = {}
        total_char = 0
        accuracy_char = 0
        predict_cum_clust_dict = predict_cum_dict[clust_key]
        frt_keys = predict_cum_clust_dict.keys()
        for frt_key in frt_keys:
            total_char = total_char + predict_cum_dict[clust_key][frt_key]['Total']
            accuracy_char = accuracy_char + predict_cum_dict[clust_key][frt_key]['Accuracy']
        predict_cum_sum_dict[clust_key]['Total'] = total_char
        predict_cum_sum_dict[clust_key]['Accuracy'] = accuracy_char
    return predict_cum_sum_dict


def summarize_prediction_accuracy(
        local_use, cluster_size, type_name, type_name_id, pred_file_name
):
    cluster_assign_arr = import_random_split_cluster(local_use, cluster_size, null_id=type_name_id)
    ###
    # cluster_assign_arr = cluster_assign_arr[:4, :]
    ###
    sub_name_length = cluster_assign_arr.shape[0]
    pred_obj_2 = {}
    pred_obj_3 = {}
    for iid in range(sub_name_length):
        target_id = cluster_assign_arr[iid, 1]
        pred_obj_2[target_id] = {}
        pred_obj_3[target_id] = {}
        pred_cluster_obj = import_swlda_obj_iid_cluster(
            local_use, target_id, type_name, type_name_id, pred_file_name
        )
        pred_cluster_obj = convert_swlda_predict_obj(pred_cluster_obj)
        # 2 compute the character-level prediction accuracy
        _, pred_cluster_obj_2 = compute_prediction_accuracy(
            local_use, 'FRT_files', target_id, pred_cluster_obj
        )
        # 3 further collapses across testing files
        pred_cluster_obj_3 = combine_prediction_accuracy(pred_cluster_obj_2)

        target_id_cluster_label = int(cluster_assign_arr[cluster_assign_arr[:, 1] == target_id, 0])
        pred_cluster_obj_2['Label'] = target_id_cluster_label
        pred_cluster_obj_3['Label'] = target_id_cluster_label
        pred_obj_2[target_id] = pred_cluster_obj_2
        pred_obj_3[target_id] = pred_cluster_obj_3
    # save pred_obj_2 and pred_obj_3
    if local_use:
        parent_path_predict = '{}/{}'.format(parent_path_local, 'FRT_files')
    else:
        parent_path_predict = '{}/{}'.format(parent_path_slurm, 'FRT_files')

    swlda_predict_dir = '{}/{}'.format(parent_path_predict, type_name)
    if not os.path.exists(swlda_predict_dir):
        os.mkdir(swlda_predict_dir)

    swlda_predict_dir = '{}/{}_dataset_{}'.format(swlda_predict_dir, type_name, type_name_id)
    if not os.path.exists(swlda_predict_dir):
        os.mkdir(swlda_predict_dir)

    with open('{}/null_predict_frt.pkl'.format(swlda_predict_dir), 'wb') as f_frt:
        pickle.dump(pred_obj_2, f_frt)
    f_frt.close()
    with open('{}/null_predict.pkl'.format(swlda_predict_dir), 'wb') as f:
        pickle.dump(pred_obj_3, f)
    f.close()
    return pred_obj_2, pred_obj_3


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


def convert_cluster_dict_to_array(cluster_assign):
    # Need to check whether cluster_assign is a dict
    cluster_size = len(cluster_assign) - 3
    cluster_id = []
    cluster_member = np.array([])
    for cluster_i in range(cluster_size):
        cluster_member_i = cluster_assign[str(cluster_i)].tolist()
        cluster_member = np.concatenate([cluster_member, cluster_member_i])
        for cluster_i_rep in range(len(cluster_member_i)):
            cluster_id.append(cluster_i)

    cluster_id = np.array([cluster_id]).T
    # cluster_id = cluster_id[:, np.newaxis]
    print(cluster_id.shape)
    # cluster_member = np.array([cluster_member])
    cluster_member = cluster_member[:, np.newaxis]
    print(cluster_member.shape)
    return np.concatenate([cluster_id, cluster_member], axis=1)


def normalized_compression_distance(x, y):
    # https: // github.com / alephmelo / pyncd / blob / master / pyncd.py
    x_y = x + y  # concatenation

    x_comp = lzma.compress(x)
    y_comp = lzma.compress(y)
    x_y_comp = lzma.compress(x_y)

    # print len() of each file
    # print(len(x_comp), len(y_comp), len(x_y_comp), sep=' ', end='\n')

    # ncd definition
    ncd = (len(x_y_comp) - min(len(x_comp), len(y_comp))) / \
          max(len(x_comp), len(y_comp))
    return ncd


def compute_bhattacharyya_distance_matrix(
        local_use, sub_name_ls, data_type, raw_file_name
):
        EEGObj2 = EEGPreFun2()
        sub_name_len = len(sub_name_ls)
        mu_tar = {}
        mu_ntar = {}
        cov_tar = {}
        cov_ntar = {}

        for sub_name_id in sub_name_ls:
            print(sub_name_id)
            eeg_TRN_dat = EEGObj2.import_trunc_eeg_obj(local_use, data_type, sub_name_id,
                                                       '{}_{}_Truncated_Data'.format(sub_name_id, raw_file_name))
            eeg_TRN_signal = eeg_TRN_dat['Signal']
            eeg_TRN_type = eeg_TRN_dat['Type']
            eeg_TRN_signal_tar = eeg_TRN_signal[eeg_TRN_type[:, 0] == 1, :]
            eeg_TRN_signal_ntar = eeg_TRN_signal[eeg_TRN_type[:, 0] == -1, :]

            mu_tar[sub_name_id] = np.mean(eeg_TRN_signal_tar, axis=0, keepdims=True).T
            mu_ntar[sub_name_id] = np.mean(eeg_TRN_signal_ntar, axis=0, keepdims=True).T
            cov_tar[sub_name_id] = np.cov(eeg_TRN_signal_tar, rowvar=False)
            cov_ntar[sub_name_id] = np.cov(eeg_TRN_signal_ntar, rowvar=False)

        distance_mat_tar = np.zeros([sub_name_len, sub_name_len])
        distance_mat_ntar = np.zeros([sub_name_len, sub_name_len])

        for i in range(sub_name_len):
            for j in range(i):
                print(i, j)
                distance_mat_tar[i, j] = compute_bhattacharyya_distance(
                    mu_tar[sub_name_ls[i]], mu_tar[sub_name_ls[j]],
                    cov_tar[sub_name_ls[i]], cov_tar[sub_name_ls[j]]
                )
                distance_mat_ntar[i, j] = compute_bhattacharyya_distance(
                    mu_ntar[sub_name_ls[i]], mu_ntar[sub_name_ls[j]],
                    cov_ntar[sub_name_ls[i]], cov_ntar[sub_name_ls[j]]
                )

        # Fill in the other half
        distance_mat_tar = (distance_mat_tar + distance_mat_tar.T) / 2
        # distance_mat_tar = distance_mat_tar / np.max(distance_mat_tar)
        distance_mat_ntar = (distance_mat_ntar + distance_mat_ntar.T) / 2
        # distance_mat_ntar = distance_mat_ntar / np.max(distance_mat_ntar)
        distance_mat = np.stack([distance_mat_tar, distance_mat_ntar], axis=0)
        save_distance_mat(local_use, distance_mat, 'bhattacharyya')
        visualize_distance_mat(local_use, distance_mat_tar, 'bhattacharyya_target', sub_name_ls)
        visualize_distance_mat(local_use, distance_mat_ntar, 'bhattacharyya_non_target', sub_name_ls)

        return np.stack([distance_mat_tar, distance_mat_ntar], axis=0)


def compute_normalized_compression_distance_matrix(
        local_use, sub_name_ls, data_type, weight_obj_name
):
    sub_name_len = len(sub_name_ls)
    b_in_model_ls = {}
    EEGObj2 = EEGPreFun2()

    for sub_name_id in sub_name_ls:
        swlda_weight_obj_id = EEGObj2.import_eeg_swlda_weights(local_use, data_type, sub_name_id, weight_obj_name)
        b_id = swlda_weight_obj_id['b']
        in_model_id = swlda_weight_obj_id['in_model']
        b_inmodel_id = np.multiply(np.transpose(in_model_id), b_id)
        b_in_model_ls[sub_name_id] = b_inmodel_id

    distance_mat = np.zeros([sub_name_len, sub_name_len])
    for i in range(sub_name_len):
        print(i)
        for j in range(sub_name_len):
            # print(i, j)
            distance_mat[i, j] = normalized_compression_distance(
                b_in_model_ls[sub_name_ls[i]], b_in_model_ls[sub_name_ls[j]]
            )

    # Fill in the other half
    distance_mat = (distance_mat + distance_mat.T) / 2
    save_distance_mat(local_use, distance_mat, 'ncd')
    visualize_distance_mat(local_use, distance_mat, 'ncd', sub_name_ls)
    return distance_mat


def save_distance_mat(local_use, mat_input, raw_file_name):
    if local_use:
        parent_path = parent_path_local
    else:
        parent_path = parent_path_slurm
    parent_path = '{}/TRN_files'.format(parent_path)
    dist_mat_dir = '{}/Dist_matrix'.format(parent_path)
    if not os.path.exists(dist_mat_dir):
        os.mkdir(dist_mat_dir)
    sio.savemat(
        '{}/{}.mat'.format(dist_mat_dir, raw_file_name),
        {'matrix': mat_input}
    )


def import_distance_mat(local_use, raw_file_name):
    if local_use:
        parent_path = parent_path_local
    else:
        parent_path = parent_path_slurm
    dist_mat_dir = '{}/TRN_files/Dist_matrix/{}.mat'.format(parent_path, raw_file_name)
    if not os.path.exists(dist_mat_dir):
        os.mkdir(dist_mat_dir)
    dist_mat_obj = sio.loadmat(dist_mat_dir)
    return dist_mat_obj['matrix']


def save_label_mat(local_use, label_arr, raw_file_name, sub_name_list):
    if local_use:
        parent_path = parent_path_local
    else:
        parent_path = parent_path_slurm
    parent_path = '{}/TRN_files'.format(parent_path)
    dist_mat_dir = '{}/Dist_matrix'.format(parent_path)
    sio.savemat(
            '{}/{}.mat'.format(dist_mat_dir, raw_file_name),
            {'label': label_arr, 'name': sub_name_list}
    )


def import_label_mat(local_use, raw_file_name):
    if local_use:
        parent_path = parent_path_local
    else:
        parent_path = parent_path_slurm
    label_mat_dir = '{}/TRN_files/Dist_matrix/{}.mat'.format(parent_path, raw_file_name)
    label_mat_obj = sio.loadmat(label_mat_dir)
    return label_mat_obj


def visualize_distance_mat(
        local_use, distance_mat, raw_file_name, sub_name_input,
        vmin=0, vmax=1
):
    if local_use:
        parent_path = parent_path_local
    else:
        parent_path = parent_path_slurm
    parent_path = '{}/TRN_files'.format(parent_path)
    dist_mat_dir = '{}/Dist_matrix'.format(parent_path)

    plot_pdf = bpdf.PdfPages('{}/{}_Plot.pdf'.format(dist_mat_dir, raw_file_name))
    # non_zero_min = 0.25
    # print(non_zero_min)
    plt.figure(figsize=(14, 12))
    ax_dist_mat = sns.heatmap(
        distance_mat,
        vmin=vmin, vmax=vmax,
        center=(vmax + vmin) / 2,
        # cmap=sns.diverging_palette(20, 220, sep=1, n=100),
        cmap=sns.color_palette("colorblind"),
        square=True
    )
    ax_dist_mat.set_xticklabels(
        # ax_dist_mat.get_xticklabels(),
        labels=sub_name_input,
        rotation=45,
        horizontalalignment='right'
    )
    ax_dist_mat.set_yticklabels(
        labels=sub_name_input,
        rotation=45,
        verticalalignment='top'
    )
    plot_pdf.savefig()
    plot_pdf.close()


def permute_distance_matrix(dist_mat, label_arr):
    n_cluster = np.max(label_arr) + 1
    permute_arr = []
    for cluster_i in range(n_cluster):
        permute_arr.extend(np.where(label_arr == cluster_i)[0])
    permute_arr = np.array(permute_arr)
    print(permute_arr.shape)
    dist_mat_2 = dist_mat[:, permute_arr]
    dist_mat_2 = dist_mat_2[permute_arr, :]
    return dist_mat_2


def permute_subject_ls(sub_name_ls, label_arr):
    n_cluster = np.max(label_arr) + 1
    permute_arr = []
    for cluster_i in range(n_cluster):
        permute_arr.extend(np.where(label_arr == cluster_i)[0])
    permute_arr = np.array(permute_arr)
    sub_name_ls = np.array(sub_name_ls)
    sub_name_ls = sub_name_ls[permute_arr]
    return sub_name_ls.tolist()


def random_split_cluster(sub_name_input, cluster_size, null_id=0):
    sub_name_length = len(sub_name_input)
    sub_ids_shuffle = np.arange(sub_name_length)
    np.random.shuffle(sub_ids_shuffle)
    # avoid empty set
    split_ids = np.random.choice(np.arange(1, sub_name_length-1), size=cluster_size-1, replace=False)
    split_ids = np.sort(split_ids)
    # print(split_ids)
    cluster_assign_list = np.array_split(sub_ids_shuffle, split_ids)
    # print(cluster_assign_list)
    # Make the output format consistent
    cluster_assign = np.zeros([sub_name_length]) - 1
    cluster_assign = cluster_assign.astype('int8')
    for clust_id, clust_group_id in enumerate(cluster_assign_list):
        cluster_assign[clust_group_id] = clust_id
    cluster_assign_arr = np.vstack([cluster_assign.tolist(), sub_name_input]).T
    return cluster_assign_arr


def save_random_split_cluster(local_use, cluster_assign_arr, cluster_size, **kwargs):
    if local_use:
        parent_path_fit = '{}/{}/Dist_matrix'.format(parent_path_local, 'TRN_files')
    else:
        parent_path_fit = '{}/{}/Dist_matrix'.format(parent_path_slurm, 'TRN_files')

    random_cluster_dir = '{}/null_{}'.format(parent_path_fit, cluster_size)
    if not os.path.exists(random_cluster_dir):
        os.mkdir(random_cluster_dir)
    if 'null_id' in kwargs.keys():
        null_id = kwargs['null_id']
    else:
        null_id = -1
    random_cluster_dir = '{}/null_{}_dataset_{}'.format(random_cluster_dir, cluster_size, null_id)
    if not os.path.exists(random_cluster_dir):
        os.mkdir(random_cluster_dir)
    np.save('{}/cluster_assign_arr.npy'.format(random_cluster_dir), cluster_assign_arr)


def import_random_split_cluster(local_use, cluster_size, **kwargs):
    if local_use:
        parent_path_fit = '{}/{}/Dist_matrix'.format(parent_path_local, 'TRN_files')
    else:
        parent_path_fit = '{}/{}/Dist_matrix'.format(parent_path_slurm, 'TRN_files')
    if 'null_id' in kwargs.keys():
        null_id = kwargs['null_id']
    else:
        null_id = -1
    random_cluster_dir = '{}/null_{}/null_{}_dataset_{}/cluster_assign_arr.npy'.format(
        parent_path_fit, cluster_size, cluster_size, null_id
    )
    cluster_assign_arr = np.load(random_cluster_dir)
    return cluster_assign_arr


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))


class ReduceDimXDAWN(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return X[:, :self.n_components, :]
