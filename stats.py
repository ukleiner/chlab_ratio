def rmspe(y_true, rmse):
    return 100*rmse/(y_true.max()-y_true.min())

def normalized_wls(param_data, wls_data, wl):
    norm_df = wls_data.divide(wls_data[wl], axis=0)
    full_norm = pd.concat([param_data, norm_df], axis=1)
    return full_norm

def calc_error(model, predict, cal=True):
    '''
    calculates the error of the prediction for each observation
    model - dict containing the T2PLS model results
    predict - what variable to predict
    cal - if using calibration or validation values
    returns:
    errors (Series)
    '''
    values = model[predict]['T2Predict']
    if cal:
        return values[0][0]-values[0][1].flatten()
    else: # validation
        return values[1][0]-values[1][1].flatten()

def ratio_ratio_r(models, key, cal=True):
    model = models[key]['Ratio']
    ratio = model['T2Predict']
    ratio_model = by_model[key]
    if cal:
        nonovo = ratio_model['cal']
        denovo = ratio[0]
    else:
        nonovo = ratio_model['val']
        denovo = ratio[1]

    return pearsonr(denovo[1].flatten(), nonovo[1])

def VIP(plsr):
    # from https://github.com/scikit-learn/scikit-learn/pull/13492/files
    T = plsr.x_scores_
    W = plsr.x_weights_
    Q = plsr.y_loadings_
    w0, w1 = W.shape
    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    s_sum = np.sum(s, axis=0)
    w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i]))
                       for i in range(w1)])
    vip = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)
    return vip
