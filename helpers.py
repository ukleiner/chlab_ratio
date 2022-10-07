def cal_val_split(df):
    cal = df['Cal_Val'] == 'Cal'
    val = df['Cal_Val'] == 'Val'
    return cal, val

def model_wls_coefs(data, coefs, filename):
    arr = np.zeros((len(coefs), 2))
    arr[:, 1] = coefs[:, 0]
    arr[:, 0] = data.vars[1]

    df = pd.DataFrame(data=arr, columns=['wl', 'coef'])
    df.to_csv(f"./coef tables/{filename}.csv", ",", index=False)

def get_line_ends(x, y):
    minloc = x.argmin()
    maxloc = x.argmax()

    return (x[minloc], x[maxloc]), (y[minloc], y[maxloc])
    # return (x.iloc[minloc], x.iloc[maxloc]), (y[minloc], y[maxloc])
