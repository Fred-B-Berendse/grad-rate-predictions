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


class LinearRegressor(Regressor):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.train_predict = None
        self.test_predict = None
        self.train_residuals = None
        self.test_residuals = None
        self.sc_coeffs = None
        self.coeffs = None
        self.means = None

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

    def _calc_scale_location(self):
        stdev_resid = np.sqrt(np.sum(self.test_residuals**2, axis=0))
        return np.sqrt(np.abs(self.test_residuals/stdev_resid))

    def plot_scale_locations(self):
        y_labels = ['sqrt(|Residual|)' for _ in self.dataset.target_labels]
        x_labels = ['Predicted '+l for l in self.dataset.target_labels]
        sc_loc = self._calc_scale_location()
        colors = [self.dataset.target_colors[l]
                  for l in self.dataset.target_labels]
        fig, ax = make_scatterplots(self.test_predict, sc_loc,
                                    x_labels=x_labels,
                                    y_labels=y_labels,
                                    colors=colors)

    def plot_coeffs_heatmap(self, normalized=False):

        X = self.model.coef_
        X = np.insert(X, 0, np.zeros(X.shape[0]), axis=1)

        if not normalized:
            scale = self.dataset.targets_scaler.scale_
            X = np.apply_along_axis(lambda r: r * scale, 0, X)
            X[:, 0] = self.dataset.targets_scaler.mean_

        y_labels = np.insert(self.dataset.feature_labels, 0, 'Intercept')
        make_heatmap(X.T, y_labels=y_labels,
                     x_labels=self.dataset.target_labels,
                     cmap='seismic_r', center=0)

    def log10_sm(self, x):
        return np.log10(x + 1)

    def log10u_sm(self, x):
        return np.log10(101-x)


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

    # # Test of capstone 2 features
    # feat_cols = np.array(['longitud', 'latitude', 'admssn_pct', 'enrlt_pct',
    #                       'enrlft_pct', 'en25', 'uagrntp', 'upgrntp',
    #                       'grntof2_pct', 'grntwf2_pct'])

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
    tr_feature_dict = {'enrlt_pct': ('log_enrlt_pct', lr.log10_sm),
                       'grntwf2_pct': ('log_grntwf2_pct', lr.log10_sm),
                       'grntof2_pct': ('log_grntof2_pct', lr.log10_sm),
                       'uagrntp': ('logu_uagrntp', lr.log10u_sm),
                       'enrlft_pct': ('logu_enrlft_pct', lr.log10u_sm)}
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
    for f, tr, te in zip(ds.target_labels, train_r2, test_r2):
        print("{} - R^2 train: {:.4f}; R^2 test: {:.4f}".format(f, tr, te))

    # Make histograms of the residuals
    lr.plot_residuals()
    plt.show()

    # Make scale-location vs predicted values plots
    lr.plot_scale_locations()
    plt.show()

    # Calculate RMSE for each target
    sc_train_rmse, sc_test_rmse = lr.rmse()
    train_rmse, test_rmse = lr.rmse(unscale=True)
    for i, f in enumerate(ds.target_labels):
        print("{}".format(f))
        formstr = "  Scaled train RMSE: {:.2f}; test RMSE: {:.2f}"
        print(formstr.format(sc_train_rmse[i], sc_test_rmse[i]))
        formstr = "  Unscaled train RMSE: {:.2f}; test RMSE: {:.2f}"
        print(formstr.format(train_rmse[i], test_rmse[i]))

    # Plot heatmap of the coefficients
    lr.plot_coeffs_heatmap(normalized=True)
    plt.show()
    lr.plot_coeffs_heatmap(normalized=False)
    plt.show()