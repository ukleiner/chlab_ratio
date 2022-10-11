import math
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
# mape is from scikit V0.24, unstable, taken from nightly build
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape

from scipy.stats import f
from scipy.signal import argrelmax

from matplotlib import lines
import matplotlib.pyplot as plt
import matplotlib.collections as collections

# Whole section based on T. Mehmood (2016) paper
# R impl: https://rdrr.io/github/khliland/plsVarSel/src/R/T2.R
def upperlimit(weights, alpha=0.0025):
    '''
    Calculate the upper Limit C(p, A*)F(A*, p-A*, alpha) for the Hotelling-T2 from a F-distribution.
    C(p, A*) = A*(p-1)/(p-A*)

    weights - matrix of loading weights with shape (p, A*)
    A* - number of components (cols)
    p - number of X variables (rows)
    returns:
        The upper limit (float)
    '''
    p, A = weights.shape
    c = A*(p-1)/(p-A)
    F = f.ppf(1-alpha, A, p-A)
    return c*F

def T2(weights):
    '''
    based on: https://github.com/cran/MSQC/blob/master/R/mult.chart.R
    calculates the Hotelling-T2 for Weights matrix

    weights - matrix of weights of dimensions p X A*
        A* is number of components
    
    returns:
        a list containing the T2 for each X variable
    '''
    inv_cov = np.linalg.pinv(np.cov(weights, rowvar=False)) # meaning variables (components) are in columns
    Wmv = weights.mean(axis=0) # Weights' mean of variable. mean val of each component's weight
    Wdiff = weights-Wmv
    # @ matrix mult in python
    t2 = [V_diff.T @ inv_cov @ V_diff for V_diff in Wdiff]
    return t2

def selectVars(weights, Xvars, alpha=1e-3):
    '''
    using Hotelling-T2 and loading weights of the PLS model choose the significant variables

    weights - loading weights of PLS model
    Xvars - vector of x variables names (wavelength numbers)
    returns: 
        selected variables names
    '''
    cutoff = upperlimit(weights, alpha)
    t2 = T2(weights)
    passedVars = t2 > cutoff
    return Xvars[passedVars]

def findOptimalLoadings(X, y, maxLoadings):
    '''
    Select the ammount of latent variables (1 to maxLoadings) that minimized PRESS
    PRESS = SIGMA(MSEcv * k) when k is n of each cv

    X - variable matrix
    y - predict variable
    maxLoadings - max number of latent variables allowed

    returns:
       number of latent variables
    '''
    scores = np.zeros((maxLoadings, 2))
    for i in range(len(scores)):
        ncomp = i+1
        pls = PLSRegression(n_components=ncomp, scale=False) # range starts from 0
        cv=10
        # ncomp, PRESS val
        scores[i] = ncomp, (cross_val_score(pls, X, y, cv=cv, scoring='neg_mean_squared_error').sum())*(len(y)/cv)
    # argrelmax instead of argelmin - looking for min PRESS but scoring returns -PRESS. So search for max -PRESS
    # argrelmax can return many minima. get the first local minima [0][0]
    try:
        min_ind = argrelmax(scores[:, 1])[0][0]
    except IndexError: # no max, 1 is best
        min_ind = 0
    return scores[min_ind, 0].astype(int)

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square(((y_true - y_pred.T[0]) / y_true)), axis=0))


def runModel(cal, val, predict, ps=None, maxLoadings=10, alpha=[0.2, 0.15, 0.1, 0.05, 0.01]):
    '''
    select the important variables for the model

    cal - calibration set
    val - validation set
    predit - column containing variable to predict
    ps - list containing X variables of model
    loadings - max ammount of loadings in the PLS model

    returns:
        dataframe with each model description
        y-predictions of the T2PLS-R
        coefficients of the T2PLS-R
        the chosen T2PLS-R model
    '''
    try: # ps might be a pandas object and fail if compared to None
        if ps == None:
            # after column 11 the wavelength variables start
            ps = cal.columns[11:]
    except:
        pass
    # R2, RMSE, is validation, is selection
    dataStore = []
    Xcal = cal[ps]
    Xval = val[ps]
    ycal = cal[predict]
    yval = val[predict]
    loadings = findOptimalLoadings(Xcal, ycal, maxLoadings)
    pls = PLSRegression(n_components = loadings, scale=False)
    pls.fit(Xcal, ycal)
    calPredict = pls.predict(Xcal)
    valPredict = pls.predict(Xval)
    dataStore.append([predict, pls.score(Xcal, ycal), mse(ycal, calPredict, squared=False), rmspe(ycal, calPredict), ps.to_numpy(), len(ps), 0, 0, None])
    
    weights = pls.x_weights_
    
    K = np.zeros((len(alpha), 8))
    V = []
    predicts = []
    coefs = []
    models = []
    for i, a in enumerate(alpha):
        selected = selectVars(weights, ps, alpha=a)
        t2_Xcal = cal[selected]
        t2_Xval = val[selected]
        if len(selected) < 2:
            continue
        t2_loadings = findOptimalLoadings(t2_Xcal, ycal, np.min([maxLoadings, len(selected)-1]))
        t2_pls = PLSRegression(n_components = t2_loadings, scale=False)
        t2_pls.fit(t2_Xcal, ycal)
        score_cal = t2_pls.score(t2_Xcal, ycal)
        yhat_cal = t2_pls.predict(t2_Xcal)
        rmse_cal = mse(ycal, yhat_cal, squared=False)
        rmspe_cal = rmspe(ycal, yhat_cal)
        
        score = t2_pls.score(t2_Xval, yval)
        yhat = t2_pls.predict(t2_Xval)
        rmse = mse(yval, yhat, squared=False)
        rmspe_val = rmspe(yval, yhat)

        K[i] = score_cal, rmse_cal, rmspe_cal,  score, rmse, rmspe_val, len(selected), a
        V.append(selected)
        predicts.append(((ycal ,yhat_cal),(yval ,yhat)))
        coefs.append((t2_pls.coef_))
        models.append(t2_pls)
    
    # minimal RMSE
    rmse_arr = K[:, 4]
    rmse_arr = rmse_arr[rmse_arr != 0]

    p = np.where(rmse_arr == np.min(rmse_arr))[0]
    ii = p[0]

    minData = K[ii]
    minVars = V[ii]
    t2predicts = predicts[ii]
    t2coefs = coefs[ii]
    t2model = models[ii]

    # cal-selection
    dataStore.append([predict, minData[0], minData[1], minData[2], minVars.to_numpy(), len(minVars), 0, 1, minData[7]])
    # val-selection
    dataStore.append([predict, minData[3], minData[4], minData[5], minVars.to_numpy(), len(minVars), 1, 1, minData[7]])
    collection = pd.DataFrame(data=dataStore, columns=['Type', 'R2', 'RMSE', 'RMSPE', 'vars', 'var_len', 'val',
                                                       'selection', 'alpha'])
    return collection, t2predicts, t2coefs, t2model

if __name__ == "__main__":
    data = pd.read_csv("20201025_Dataset_plots_n216.csv")
    # remove bad entries
    cleanData = data[(data['NotAnalizedFor'].isna())]
    # drop noisy wavelengths
    cleanData = cleanData.drop(columns=map(str, range(329, 400)))
    # split cal-val as in previous analysis
    calData = cleanData[cleanData['Cal_Val'] == 'Cal']
    valData = cleanData[cleanData['Cal_Val'] == 'Val']

    # only using Flag leaf data
    flagLeafData = cleanData[cleanData['LeafType'] == 'FlagLeaf']
    calFlag = flagLeafData[flagLeafData['Cal_Val'] == 'Cal']
    valFlag = flagLeafData[flagLeafData['Cal_Val'] == 'Val']

    wls_columns = cleanData.columns[11:]

    #Ca, Cb, TChl
    CaData, CaT2Predict, CaCoefs, CaModels = runModel(calData, valData, 'Ca')
    CbData, CbT2Predict, CbCoefs, CbModels = runModel(calData, valData, 'Cb')
    TChlData, TChlT2Predict, TChlCoefs, TChlModels = runModel(calData, valData, 'TChl')

    #Ca, Cb, TChl for FlagLeaf data only
    Flag_CaData, Flag_CaT2Predict, Flag_CaCoefs, Flag_CaModels = runModel(calFlag, valFlag, 'Ca')
    Flag_CbData, Flag_CbT2Predict, Flag_CbCoefs, Flag_CbModels = runModel(calFlag, valFlag, 'Cb')
    Flag_TChlData, Flag_TChlT2Predict, Flag_TChlCoefs, Flag_TChlModels = runModel(calFlag, valFlag, 'TChl')
    Flag_CaData['Type'] += 'Flag'
    Flag_CbData['Type'] += 'Flag'
    Flag_TChlData['Type'] += 'Flag'

    #Ca, Cb, Tchl for normalized 525wl data only
    norm525_wl = cleanData[wls_columns].apply(lambda row: row/row['525'], axis=1)
    Ca525Data, Ca525T2Predict, Ca525Coefs, Ca525Models = runModel(calData, valData, 'Ca')
    Cb525Data, CbT2Predict, Cb525Coefs, Cb525Models = runModel(calData, valData, 'Cb')
    TChl525Data, TChl525T2Predict, TChl525Coefs, TChl525Models = runModel(calData, valData, 'TChl')
