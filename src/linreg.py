import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from plotting import make_histograms, make_scatterplots, make_heatmap
from dataset import Dataset
from regressor import Regressor


def log10_sm(x):
    return np.log10(x + 1)


def log10u_sm(x):
    return np.log10(101-x)


def variance_inflation_factors(Xarr):
    vifs = np.array([], dtype=float)
    for i in range(Xarr.shape[1]):
        X = Xarr.copy()
        y = X[:, i]
        X = np.delete(X, i, axis=1)
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        r_sq = reg.score(X, y)
        vif = 1.0/(1.0-r_sq) if r_sq < 1.0 else np.inf
        vifs = np.append(vifs, vif)
    return vifs


class LinearRegressor(Regressor):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.train_predict = None
        self.test_predict = None
        self.train_residuals = None
        self.test_residuals = None

    def calc_vifs(self):
        Xarr = self.dataset.X_train
        vifs = np.array([], dtype=float)
        for i in range(Xarr.shape[1]):
            X = Xarr.copy()
            y = X[:, i]
            X = np.delete(X, i, axis=1)
            reg = LinearRegression(n_jobs=-1).fit(X, y)
            r_sq = reg.score(X, y)
            vif = 1.0/(1.0-r_sq) if r_sq < 1.0 else np.inf
            vifs = np.append(vifs, vif)
        return vifs

    def _calc_qq(self, target):
        pass

    def plot_residual_qq(self, targets_list=None):
        '''
        plots the qq plot of each residual in targets_list
        if targets_list is none, plot all targets
        '''
        pass

    def _calc_scale_location(self, target):
        pass

    def plot_scale_locations(self, target_list):
        pass

    def plot_coeffs_heatmap(features, targets, standardized=False):
        '''
        if normalized=True, then standardized coefficients are graphed
        otherwise regular coefficients are graphed
        '''
        pass


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')
    mdf.drop(['Unnamed: 0', 'applcn'], axis=1, inplace=True)

    # Original features
    # feat_cols = ['iclevel_2to4', 'iclevel_0to2', 'iclevel_na',
    #              'control_public', 'control_privnp', 'control_na',
    #              'hloffer_assoc', 'hloffer_doct', 'hloffer_bach',
    #              'hloffer_mast', 'hloffer_2to4yr', 'hloffer_0to1yr',
    #              'hloffer_postmc', 'hloffer_na', 'hloffer_postbc',
    #              'hbcu_yes', 'tribal_yes', 'locale_ctylrg', 'locale_ctysml',
    #              'locale_ctymid', 'locale_twndst', 'locale_rurfrg',
    #              'locale_twnrem', 'locale_submid', 'locale_subsml',
    #              'locale_twnfrg', 'locale_rurdst', 'locale_rurrem',
    #              'locale_na', 'instsize_1to5k', 'instsize_5to10k',
    #              'instsize_10to20k', 'instsize_na', 'instsize_gt20k',
    #              'instsize_norpt', 'landgrnt_yes', 'longitud', 'latitude',
    #              'admssn_pct', 'enrlt_pct', 'enrlft_pct', 'en25', 'en75',
    #              'mt25', 'mt75', 'uagrntp', 'upgrntp', 'npgrn2', 'grnton2_pct',
    #              'grntof2_pct', 'grntwf2_pct']

    # Surviving features after VIF elimination
    feat_cols = ['control_privnp', 'hloffer_postmc', 'hloffer_postbc',
                 'hbcu_yes', 'locale_ctylrg', 'locale_ctysml',
                 'locale_ctymid', 'locale_twndst', 'locale_rurfrg',
                 'locale_twnrem', 'locale_submid', 'locale_subsml',
                 'locale_twnfrg', 'locale_rurdst', 'locale_rurrem',
                 'instsize_1to5k', 'instsize_5to10k', 'instsize_10to20k',
                 'instsize_gt20k', 'longitud', 'latitude',
                 'admssn_pct', 'enrlt_pct', 'enrlft_pct',
                 'en25', 'uagrntp', 'upgrntp',
                 'npgrn2', 'grntof2_pct', 'grntwf2_pct']

    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
                            'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
                            'cstcball_pct_grwhitt', 'pgcmbac_pct',
                            'sscmbac_pct', 'nrcmbac_pct'])

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.scale_features_targets()

    lr = LinearRegressor(LinearRegression(), ds)
    lr.dataset = ds

    vifs = lr.calc_vifs()
    print(f"VIF: {vifs}")

    # Make histograms of the features
    lr.make

    # transform some features
    # tr_feature_dict = {'enrlt_pct': ('log_enrlt_pct', log10_sm),
    #                    'grntwf2_pct': ('log_grntwf2_pct', log10_sm),
    #                    'grntof2_pct': ('log_grntof2_pct', log10_sm),
    #                    'uagrntp': ('logu_uagrntp', log10u_sm),
    #                    'enrlft_pct': ('logu_enrlft_pct', log10u_sm)}

    # ds.transform_features(tr_feature_dict)
    # ds.scale_features_targets()

    # Variance Inflation Factors
    # # removed through recursive VIF:
    # #       'iclevel', 'tribal', 'grnton2_pct', 'mt25', 'en75', 'mt75',

    # # Variance Inflation Factors
    # vif = variance_inflation_factors(X_train_sc)
    # print(f"VIF: {vif}")

    # # Make histogram of the graduation rates
    # x_labels = ['Grad. Rate: ' + l for l in labels]
    # fig, ax = make_histograms(Y_train, x_labels=x_labels, colors=colors)
    # plt.show()

    # # Make histograms of the features
    # fig, ax = make_histograms(X_train, x_labels=feat_cols)
    # plt.show()

    # # Fit the training and test data
    # reg = LinearRegression().fit(X_train_sc, Y_train_sc)
    # X_test_sc = xsc.transform(X_test)
    # Y_test_sc = ysc.transform(Y_test)

    # for i, l in enumerate(labels):
    #     print(l)
    #     reg2 = LinearRegression().fit(X_train_sc, Y_train_sc[:, i])
    #     print(f"Training R^2: {reg2.score(X_train_sc, Y_train_sc[:, i])}")
    #     print(f"Test R^2: {reg2.score(X_test_sc, Y_test_sc[:, i])}")
    
    # # Make histogram of the residuals
    # Y_pred_sc = reg.predict(X_test_sc)
    # Y_resid_sc = Y_test_sc - Y_pred_sc
    # x_labels = ['Residuals: ' + l for l in labels]
    # fig, ax = make_histograms(Y_resid_sc, x_labels=x_labels,
    #                           colors=colors, center=0)
    # plt.show()

    # # Plot scale-location plot vs predicted values
    # # Scale-location plot is sqrt(abs(resid)/std(residuals))
    # stdev_resid = np.sqrt(np.sum(Y_resid_sc**2, axis=0))
    # std_Y_resid_sc = np.sqrt(np.abs(Y_resid_sc/stdev_resid))
    # y_labels = ['sqrt(|Residual|)' for _ in labels]
    # x_labels = ['Predicted Norm. Rate: ' + l for l in labels]
    # fig, ax = make_scatterplots(Y_pred_sc, std_Y_resid_sc, x_labels=x_labels,
    #                             y_labels=y_labels, colors=colors)
    # plt.show()

    # # Calculate RMSE for each target
    # rmse_sc = np.sqrt((Y_resid_sc**2).sum(axis=0)/Y_resid_sc.shape[0])
    # rmse = rmse_sc*ysc.scale_
    # for k, v in zip(labels, rmse):
    #     print(f"RMSE for {k}: {v}")

    # coeffs_sc = reg.coef_
    # coeffs_sc = np.insert(coeffs_sc, 0, np.zeros(coeffs_sc.shape[0]), axis=1)
    # x_labels = np.insert(feat_cols, 0, 'Intercept')
    # ax = make_heatmap(coeffs_sc, x_labels=x_labels, y_labels=labels,
    #                   cmap='seismic_r', center=0)
    # plt.show()

    # coeffs = np.apply_along_axis(lambda r: r*ysc.scale_, 0, coeffs_sc)   
    # coeffs[:, 0] = ysc.mean_
    # ax = make_heatmap(coeffs, x_labels=x_labels, y_labels=labels,
    #                   cmap='seismic_r', center=0)
    # plt.show()

