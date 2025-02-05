from self_py_fun.Misc import *
from pyriemann.transfer import MDWM
from pyriemann.estimation import ERPCovariances
from sklearn.pipeline import make_pipeline
from pyriemann.spatialfilters import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from self_py_fun.PreFun2 import Vectorizer, ReduceDimXDAWN


def convert_y_label(y_input, domain_type):
    r"""
    :param y_input:
    :param domain_type:
    :return:
    """
    y_input_str = []
    for y_input_iter in y_input:
        y_input_str.append('{}/{}'.format(domain_type, int(y_input_iter)))
    y_input_str = np.array(y_input_str)
    return y_input_str


def create_x_cont_y_eeg_mdwm(X_input, domain_type):
    r"""
    :param X_input:
    :param domain_type:
    :return:
    """
    X_input_cont = np.concatenate([X_input['target'], X_input['non-target']], axis=0)
    y_input_cont = np.concatenate([np.ones(X_input['target'].shape[0]),
                                   -np.ones(X_input['non-target'].shape[0])], axis=0)
    y_new_mdwm = convert_y_label(y_input_cont, domain_type)

    return X_input_cont, y_new_mdwm


def create_x_cont_y_sim_mdwm(data_input, domain_type, seq_size_train=1, **kwargs):
    r"""
    :param data_input:
    :param domain_type:
    :param seq_size_train:
    :return:
    """

    X_input = np.array(data_input['X'])
    y_input = np.array(data_input['Y'])
    if 'E' in kwargs.keys() and kwargs['E'] == 1:
        char_size, seq_size, rcp_size, signal_len_size = X_input.shape
        channel_size = 1
    else:
        char_size, seq_size, rcp_size, channel_size, signal_len_size = X_input.shape
    if seq_size_train is not None and seq_size_train < seq_size:
        X_input = X_input[:, :seq_size_train, ...]
        y_input = y_input[:, :seq_size_train, :]
        seq_size = seq_size_train
    X_input_cont = np.reshape(X_input, [char_size * seq_size * rcp_size, channel_size, signal_len_size])
    y_input_cont = np.reshape(y_input, [char_size * seq_size * rcp_size])
    y_new_mdwm = convert_y_label(y_input_cont, domain_type)

    return X_input_cont, y_new_mdwm


def mdwm_fit_from_sim_feature(
        sim_data, sub_new_name, source_sub_name_ls, domain_tradeoff, tol, **kwargs
):
    X_domain = []
    y_domain = []
    new_domain = 'new_domain'

    new_data = sim_data[sub_new_name]
    X_new_cont, y_new_mdwm = create_x_cont_y_sim_mdwm(new_data, new_domain, **kwargs)
    X_domain.append(X_new_cont)
    y_domain.append(y_new_mdwm)

    source_domain = 'source_domain'
    for source_sub_name_iter in source_sub_name_ls:
        X_domain_sub = sim_data[source_sub_name_iter]
        X_source_sub_cont, y_source_sub_mdwm = create_x_cont_y_sim_mdwm(X_domain_sub, source_domain, **kwargs)

        X_domain.append(X_source_sub_cont)
        y_domain.append(y_source_sub_mdwm)

    X_domain = np.concatenate(X_domain, axis=0)
    y_domain = np.concatenate(y_domain, axis=0)

    ERP_cov_obj = ERPCovariances(estimator='scm')
    ERP_cov_obj.fit(X_domain, y_domain)
    X_domain_cov = ERP_cov_obj.transform(X_domain)
    X_domain_cov_size = X_domain_cov.shape[1]
    X_domain_cov = X_domain_cov + np.eye(X_domain_cov_size) * tol

    # domain_tradeoff = 0.5
    # target_train_frac = 0.1
    # domain_tradeoff = 1 - np.exp(-25 * target_train_frac)
    mdwm_obj = MDWM(
        domain_tradeoff=domain_tradeoff, metric='riemann',
        target_domain=new_domain
    )
    mdwm_obj.fit(X_domain_cov, y_domain)

    pred_score = np.log(mdwm_obj.predict_proba(X_domain_cov)[:, 0])

    mu_tar = np.mean(pred_score[np.array(['1' in y_domain_iter for y_domain_iter in y_domain])])
    mu_ntar = np.mean(pred_score[np.array(['0' in y_domain_iter for y_domain_iter in y_domain])])
    std_common = np.std(pred_score)

    return [mdwm_obj, ERP_cov_obj, X_domain_cov_size,
            {'mu_tar': mu_tar, 'mu_ntar': mu_ntar, 'std_common': std_common}]


def mdwm_fit_from_eeg_feature(
        new_data, source_data, source_sub_name_ls, domain_tradeoff, tol
):
    r"""
    :param new_data:
    :param source_data:
    :param source_sub_name_ls:
    :param domain_tradeoff:
    :param tol:
    :return:
    """

    X_domain = []
    y_domain = []
    new_domain = 'new_domain'
    X_new_cont, y_new_mdwm = create_x_cont_y_eeg_mdwm(new_data, new_domain)
    X_domain.append(X_new_cont)
    y_domain.append(y_new_mdwm)

    source_domain = 'source_domain'
    for source_sub_name_iter in source_sub_name_ls:
        X_domain_sub = source_data[source_sub_name_iter]
        X_source_sub_cont, y_source_sub_mdwm = create_x_cont_y_eeg_mdwm(X_domain_sub, source_domain)

        X_domain.append(X_source_sub_cont)
        y_domain.append(y_source_sub_mdwm)

    X_domain = np.concatenate(X_domain, axis=0)
    y_domain = np.concatenate(y_domain, axis=0)

    ERP_cov_obj = ERPCovariances(estimator='scm')
    ERP_cov_obj.fit(X_domain, y_domain)
    X_domain_cov = ERP_cov_obj.transform(X_domain)
    X_domain_cov_size = X_domain_cov.shape[1]
    X_domain_cov = X_domain_cov + np.eye(X_domain_cov_size) * tol

    # domain_tradeoff = 0.5
    # target_train_frac = 0.1
    # domain_tradeoff = 1 - np.exp(-25 * target_train_frac)
    mdwm_obj = MDWM(
        domain_tradeoff=domain_tradeoff, metric='riemann',
        target_domain=new_domain
    )
    mdwm_obj.fit(X_domain_cov, y_domain)

    pred_score = np.log(mdwm_obj.predict_proba(X_domain_cov)[:, 0])

    mu_tar = np.mean(pred_score[np.array(['-1' not in y_domain_iter for y_domain_iter in y_domain])])
    mu_ntar = np.mean(pred_score[np.array(['-1' in y_domain_iter for y_domain_iter in y_domain])])
    std_common = np.std(pred_score)

    return [mdwm_obj, ERP_cov_obj, X_domain_cov_size,
            {'mu_tar': mu_tar, 'mu_ntar': mu_ntar, 'std_common': std_common}]


def predict_xdawn_lda_fast_multi(
        n_components, xdawn_min, new_signal, new_label
):
    clf = make_pipeline(
        Xdawn(nfilter=n_components, estimator='scm'),
        ReduceDimXDAWN(n_components=xdawn_min),
        Vectorizer(),
        LDA(solver="lsqr", shrinkage="auto"),
    )
    clf.fit(new_signal, new_label)

    return clf


