import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV
from colors import targets_color_dict, get_colors
from dataset import Dataset
from linreg import LinearRegressor


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

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.target_labels = ['2+ Races', 'Asian', 'Black', 'Hispanic', 'White',
                        'Pell Grant', 'SSL', 'Non-Recipient']
    ds.target_colors = targets_color_dict()

    model = MultiTaskLassoCV(cv=5, fit_intercept=False, n_jobs=-1)
    lr = LinearRegressor(model, ds)

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

    # Make histograms of the residuals
    lr.plot_residuals()
    plt.show()

    # Make scale-location vs predicted values plots
    lr.plot_scale_locations()
    plt.show()

    # Calculate R^2 and RMSE for each target
    train_r2, test_r2 = lr.r_squared()
    sc_train_rmse, sc_test_rmse = lr.rmse()
    train_rmse, test_rmse = lr.rmse(unscale=True)
    for i, f in enumerate(ds.target_labels):
        print("{}".format(f))
        formstr = "  train R^2: {:.3f}; test R^2: {:.3f}"
        print(formstr.format(train_r2[i], test_r2[i]))
        formstr = "  Scaled train RMSE: {:.2f}; test RMSE: {:.2f}"
        print(formstr.format(sc_train_rmse[i], sc_test_rmse[i]))
        formstr = "  Unscaled train RMSE: {:.2f}; test RMSE: {:.2f}"
        print(formstr.format(train_rmse[i], test_rmse[i]))

    # Plot heatmap of the coefficients
    lr.plot_coeffs_heatmap(normalized=True)
    plt.show()
    lr.plot_coeffs_heatmap(normalized=False)
    plt.show()

    # Print statistics for each test
    print("alpha: {:.5f}".format(lr.model.alpha_))
    print("n_iterations: {}".format(lr.model.n_iter_))
