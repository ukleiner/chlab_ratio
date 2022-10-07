# Whole section based on T. Mehmood (2016) paper
# R impl: https://rdrr.io/github/khliland/plsVarSel/src/R/T2.R
def upperlimit(weights, alpha=0.0025):
    '''
    Calculate the upper Limit C(p, A*)F(A*, p-A*, alpha) based on Hotteling's-T2
    C(p, A*) = A*(p-1)/(p-A*)
    weights - matrix of loading weights with shape (p, A*)
    A* - number of components (cols)
    p - number of X variables (rows)
    '''
    p, A = weights.shape
    c = A*(p-1)/(p-A)
    F = f.ppf(1-alpha, A, p-A)
    return c*F

def T2(weights):
    '''
    based on: https://github.com/cran/MSQC/blob/master/R/mult.chart.R
    calculates the Hotteling's-T2 for Weights matrix
    weights - matrix of weights of dimensions p X A*
    A* is number of components

    returns a list containing the T2 for each X variable
    '''
    inv_cov = np.linalg.pinv(np.cov(weights, rowvar=False)) # meaning variables (components) are in columns
    Wmv = weights.mean(axis=0) # Weights' mean of variable. mean val of each component's weight
    Wdiff = weights-Wmv
    t2 = [V_diff.T @ inv_cov @ V_diff for V_diff in Wdiff]
    return t2

def selectVars(weights, Xvars, alpha=1e-3):
    '''
    using Hotteling's-T2 and loading weights of the PLS model choose the significant variables
    weights - loading weights of PLS model
    Xvars - dataframe of x variables (wavelengths)
    returns:
    selected variables dataframe
    '''
    cutoff = upperlimit(weights, alpha)
    t2 = T2(weights)
    passedVars = t2 > cutoff
    return Xvars.loc[:, passedVars]
