import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from plotting import make_histograms, make_scatterplots, make_heatmap
from plotting import make_color_dict
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
    #              'mt25', 'mt75', 'uagrntp', 'upgrntp', 'npgrn2', 
    #              'grnton2_pct', 'grntof2_pct', 'grntwf2_pct']

    # Surviving features after VIF elimination
    feat_cols = np.array(['control_privnp', 'hloffer_postmc', 'hloffer_postbc',
                          'hbcu_yes', 'locale_ctylrg', 'locale_ctysml',
                          'locale_ctymid', 'locale_twndst', 'locale_rurfrg',
                          'locale_twnrem', 'locale_submid', 'locale_subsml',
                          'locale_twnfrg', 'locale_rurdst', 'locale_rurrem',
                          'instsize_1to5k', 'instsize_5to10k',
                          'instsize_10to20k', 'instsize_gt20k', 'longitud',
                          'latitude', 'admssn_pct', 'enrlt_pct', 'enrlft_pct',
                          'en25', 'uagrntp', 'upgrntp', 'npgrn2',
                          'grntof2_pct', 'grntwf2_pct'])

    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
                            'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
                            'cstcball_pct_grwhitt', 'pgcmbac_pct',
                            'sscmbac_pct', 'nrcmbac_pct'])

    labels = ['2+ Races', 'Asian', 'Black', 'Hispanic', 'White', 'Pell Grant',
              'SSL', 'Non-Recipient']

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.target_labels = np.array(['Graduation Rate: ' + l for l in labels])

    # Assign colors to the dataset
    labels = ['Asian', 'Black', 'Hispanic', 'Nat. Am.', 'Pac. Isl.', 'White',
              '2+ Races']
    labels = np.array(['Graduation Rate: ' + l for l in labels]) 
    color_dict = make_color_dict(labels, cm.Accent)
    labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    labels = np.array(['Graduation Rate: ' + l for l in labels])
    color_dict.update(make_color_dict(labels, cm.brg))
    ds.target_colors = color_dict

    # Calculate variance inflation factors
    lr = LinearRegressor(LinearRegression(), ds)
    vifs = lr.calc_vifs()
    print(f"VIF: {vifs}")

    # transform some features
    tr_feature_dict = {'enrlt_pct': ('log_enrlt_pct', log10_sm),
                       'grntwf2_pct': ('log_grntwf2_pct', log10_sm),
                       'grntof2_pct': ('log_grntof2_pct', log10_sm),
                       'uagrntp': ('logu_uagrntp', log10u_sm),
                       'enrlft_pct': ('logu_enrlft_pct', log10u_sm)}
    ds.transform_features(tr_feature_dict, drop_old=False)

    # Plot histograms of transformed features
    features = ['enrlt_pct', 'log_enrlt_pct',
                'enrlft_pct', 'logu_enrlft_pct',
                'uagrntp', 'logu_uagrntp',
                'grntwf2_pct', 'log_grntwf2_pct',
                'grntof2_pct', 'log_grntof2_pct']
    x_labels = ['Percent Enrolled', 'Log Percent Enrolled',
                'Percent Full Time', 'Log (100 - Percent Full Time)',
                'Percent Awarded Aid (any)',
                'Log (100 - Percent Awarded Aid (any))',
                'Percent With Family: 2016-17',
                'Log Percent With Family: 2016-17',
                'Percent Off Campus: 2016-17',
                'Log Percent Off Campus: 2016-17']
    ds.make_feature_histograms(features, x_labels=x_labels)
    plt.show()

    features = ['enrlt_pct', 'grntwf2_pct', 'grntof2_pct', 'uagrntp',
                'enrlft_pct']
    ds.drop_features(features)
    ds.scale_features_targets()

    # Perform fitting and predicting
    lr.fit_train()
    lr.predict()

    # Calculate training and test scores
    train_r2, test_r2 = lr.r_squared()
    print("R^2 train: {:.4f}; R^2 test: {:.4f}".format(train_r2, test_r2))

    # Make histograms of the residuals
    lr.plot_residuals()
    plt.show()

    # Make scale-location vs predicted values plots
    
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

