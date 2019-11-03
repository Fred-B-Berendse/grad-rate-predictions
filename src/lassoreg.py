import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from linreg import make_histograms, make_scatterplots, make_heatmap


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_model.csv')
    mdf.drop(['Unnamed: 0', 'applcn'], axis=1, inplace=True)

    mdf['log_enrlt_pct'] = np.log10(mdf['enrlt_pct'])
    mdf['log_grntwf2_pct'] = np.log10(mdf['grntwf2_pct']+1)
    mdf['log_grntof2_pct'] = np.log10(mdf['grntof2_pct']+1)
    mdf['logu_uagrntp'] = np.log10(101-mdf['uagrntp'])
    mdf['logu_grntof2_pct'] = np.log10(101-mdf['grntof2_pct'])
    mdf['logu_enrlft_pct'] = np.log10(101-mdf['enrlft_pct'])

    feat_cols = np.array(['longitud', 'latitude', 'logu_enrlft_pct',
                          'log_enrlt_pct', 'admssn_pct', 'en25',
                          'logu_uagrntp', 'upgrntp', 'log_grntwf2_pct',
                          'logu_grntof2_pct'])

    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
                            'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
                            'cstcball_pct_grwhitt', 'pgcmbac_pct',
                            'sscmbac_pct', 'nrcmbac_pct'])
    race_labels = ['Two or More Races', 'Asian', 'Black', 'Hispanic', 'White']
    race_colors = cm.Accent(np.linspace(0, 1, len(race_labels)))
    pgs_labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    pgs_colors = cm.Set2(np.linspace(0, 1, len(pgs_labels)))
    labels = np.append(race_labels, pgs_labels)
    colors = np.append(race_colors, pgs_colors, axis=0)

    # Create a train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
            mdf.loc[:, feat_cols].values,
            mdf.loc[:, target_cols].values, random_state=10)

    # scale the training and test sets
    xsc = StandardScaler()
    X_train_sc = xsc.fit_transform(X_train)
    X_test_sc = xsc.transform(X_test)
    ysc = StandardScaler()
    Y_train_sc = ysc.fit_transform(Y_train)
    Y_test_sc = ysc.transform(Y_test)

    # Fit the training and test data
    regarr = []
    Y_pred_sc = np.array([])
    for i in range(len(target_cols)):
        print(f"target: {target_cols[i]}")
        reg = LassoLarsCV(cv=5, fit_intercept=False, n_jobs=-1)
        x_train, y_train = X_train_sc, Y_train_sc[:, i]
        reg.fit(x_train, y_train)
        print(f"mean y_train = {y_train.mean()}")
        print(f"train R^2 = {reg.score(x_train, y_train)}")
        x_test, y_test = X_test_sc, Y_test_sc[:, i]
        print(f"test R^2 = {reg.score(x_test, y_test)}")
        print(f"mean y_test = {y_test.mean()}")
        regarr.append(reg)
        y_pred = reg.predict(x_test)
        Y_pred_sc = np.append(Y_pred_sc, y_pred)
    Y_pred_sc = Y_pred_sc.reshape(Y_test_sc.shape)

    # Make histogram of the residuals
    Y_resid_sc = Y_test_sc - Y_pred_sc
    x_labels = ['Residuals: ' + l for l in labels]
    fig, ax = make_histograms(Y_resid_sc, x_labels=x_labels,
                              colors=colors, center=0)
    plt.show()

    # Plot scale-location plot vs predicted values
    # Scale-location plot is sqrt(abs(resid)/std(residuals))
    stdev_resid = np.sqrt(np.sum(Y_resid_sc**2, axis=0))
    std_Y_resid_sc = np.sqrt(np.abs(Y_resid_sc/stdev_resid))
    y_labels = ['sqrt(|Residual|)' for _ in labels]
    x_labels = ['Predicted Norm. Rate: ' + l for l in labels]
    fig, ax = make_scatterplots(Y_pred_sc, std_Y_resid_sc, x_labels=x_labels,
                                y_labels=y_labels, colors=colors)
    plt.show()

    # Calculate RMSE for each target
    rmse_sc = np.sqrt((Y_resid_sc**2).sum(axis=0)/Y_resid_sc.shape[0])
    rmse = rmse_sc*ysc.scale_
    for k, v in zip(labels, rmse):
        print(f"RMSE for {k}: {v}")

    # Plot the coefficients heatmaps
    coeffs_sc = np.array([])
    for reg in regarr:
        coeffs_sc = np.append(coeffs_sc, reg.coef_)
    coeffs_sc = coeffs_sc.reshape(len(regarr), len(feat_cols))
    coeffs_sc = np.insert(coeffs_sc, 0, np.zeros(coeffs_sc.shape[0]), axis=1)
    x_labels = np.insert(feat_cols, 0, 'Intercept')
    ax = make_heatmap(coeffs_sc, x_labels=x_labels, y_labels=labels,
                      cmap='seismic_r', center=0)
    plt.show()

    coeffs = np.apply_along_axis(lambda r: r*ysc.scale_, 0, coeffs_sc)
    coeffs[:, 0] = ysc.mean_
    ax = make_heatmap(coeffs, x_labels=x_labels, y_labels=labels,
                      cmap='seismic_r', center=0)
    plt.show()

    # Print statistics for each test
    for i, reg in enumerate(regarr):
        print(f"target: {labels[i]}")
        print(f"  alpha: {reg.alpha_}")
        print(f"  n_itr: {reg.n_iter_}")
