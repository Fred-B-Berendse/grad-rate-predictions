import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import Dataset
from colors import targets_color_dict
import pymc3 as pm
from regressor import Regressor
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


class McmcRegressor(Regressor):

    def __init__(self, dataset):
        self.dataset = dataset
        self.trace = None
        self.train_predict = None
        self.test_predict = None
        self.train_residuals = None
        self.test_residuals = None
        self.sc_coeffs = None
        self.coeffs = None
        self.means = None

    @staticmethod
    def _replace_bad_chars(mystr):
        res = mystr.replace(' ', '_').replace('+', 'pl')
        res = res.replace('-', '_')
        return res

    def _make_formula(self, feat_list, target_label):
        tl = self._replace_bad_chars(target_label)
        return tl + ' ~ ' + ' + '.join(feat_list)

    def _format_data(self, target_label):
        data = pd.DataFrame(data=self.dataset.X_train,
                            columns=self.dataset.feature_labels)
        tar_col = self._replace_bad_chars(target_label)
        idx = np.argwhere(self.dataset.target_labels == target_label)[0][0]
        data[tar_col] = self.dataset.Y_train[:, idx]
        return data

    def build_model(self, target_label, draws=2000, tune=500):

        with pm.Model() as model:
            data = self._format_data(target_label)
            print("data columns: {}".format(data.columns))
            formula = self._make_formula(self.dataset.feature_labels,
                                         target_label)
            print("formula: {}".format(formula))
            # family = pm.glm.families.Normal()
            pm.glm.GLM.from_formula(formula, data=data)
            self.trace = pm.sample(draws=draws, tune=tune, cores=4)


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')
    mdf.drop(['Unnamed: 0', 'applcn'], axis=1, inplace=True)

    # Surviving features after VIF elimination
    # feat_cols = np.array(['control_privnp', 'hloffer_postmc', 'hloffer_postbc',
    #                       'hbcu_yes', 'locale_ctylrg', 'locale_ctysml',
    #                       'locale_ctymid', 'locale_twndst', 'locale_rurfrg',
    #                       'locale_twnrem', 'locale_submid', 'locale_subsml',
    #                       'locale_twnfrg', 'locale_rurdst', 'locale_rurrem',
    #                       'instsize_1to5k', 'instsize_5to10k',
    #                       'instsize_10to20k', 'instsize_gt20k', 'longitud',
    #                       'latitude', 'admssn_pct', 'enrlt_pct', 'enrlft_pct',
    #                       'en25', 'uagrntp', 'upgrntp',
    #                       'npgrn2', 'grntof2_pct', 'grntwf2_pct'])

    # feat_cols = np.array(['en25', 'upgrntp', 'latitude', 'control_privnp',
    #                       'locale_twnrem', 'locale_rurrem', 'grntof2_pct',
    #                       'uagrntp', 'enrlt_pct'])

    feat_cols = np.array(['en25', 'upgrntp'])

    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
                            'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
                            'cstcball_pct_grwhitt', 'pgcmbac_pct',
                            'sscmbac_pct', 'nrcmbac_pct'])

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.target_labels = np.array(['2+ Races', 'Asian', 'Black', 'Hispanic',
                                 'White', 'Pell Grant', 'SSL',
                                 'Non-Recipient'])
    ds.target_colors = targets_color_dict()
    # tr_feature_dict = {'grntof2_pct': ('log_grntof2_pct', ds.log10_sm),
    #                    'uagrntp': ('logu_uagrntp', ds.log10u_sm),
    #                    'enrlt_pct': ('log_enrlt_pct', ds.log10_sm)}
    # ds.transform_features(tr_feature_dict, drop_old=True)

    mcmc = McmcRegressor(ds)
    mcmc.build_model('Non-Recipient')

    # transform some features
    # tr_feature_dict = {'enrlt_pct': ('log_enrlt_pct', lr.log10_sm),
    #                    'grntwf2_pct': ('log_grntwf2_pct', lr.log10_sm),
    #                    'grntof2_pct': ('log_grntof2_pct', lr.log10_sm),
    #                    'uagrntp': ('logu_uagrntp', lr.log10u_sm),
    #                    'enrlft_pct': ('logu_enrlft_pct', lr.log10u_sm)}

    # ds.scale_features_targets()

    # with pm.Model() as model:
    #     alpha = pm.Normal('alpha', 0, 20)
    #     beta0 = pm.Normal('beta_en25', 0, 10)
    #     beta1 = pm.Normal('beta_upgrntp', 0, 10)
    #     beta2 = pm.Normal('beta_latitude', 0, 10)
    #     beta3 = pm.Normal('beta_control_privnp', 0, 10)
    #     beta4 = pm.Normal('beta_locale_twnrem', 0, 10)
    #     beta5 = pm.Normal('beta_locale_rmrrem', 0, 10)
    #     beta6 = pm.Normal('beta_log_grntof2_pct', 0, 10)
    #     beta7 = pm.Normal('beta_logu_uagrntp', 0, 10)
    #     beta8 = pm.Normal('beta_log_enrlt_pct', 0, 10)
    #     sigma = pm.HalfNormal('sigma', 10)
    #     # mu = alpha + beta*ds.X_train
    #     mu = alpha + beta0*ds.X_train[:, 0] + beta1*ds.X_train[:, 1] + \
    #         beta2 * ds.X_train[:, 2] + beta3 * ds.X_train[:, 3] + \
    #         beta4 * ds.X_train[:, 4] + beta5 * ds.X_train[:, 5] + \
    #         beta6 * ds.X_train[:, 6] + beta7 * ds.X_train[:, 7] + \
    #         beta8 * ds.X_train[:, 8]  # + beta[9] * ds.X_train[:, 9] + \
    #         #  beta[10] * ds.X_train[:, 10]
    #         #  beta[10] * ds.X_train[:, 10] + beta[11] * ds.X_train[:, 11] + \
    #         #  beta[12] * ds.X_train[:, 12] + beta[13] * ds.X_train[:, 13] + \
    #         #  beta[14] * ds.X_train[:, 14] + beta[15] * ds.X_train[:, 15] + \
    #         #  beta[16] * ds.X_train[:, 16] + beta[17] * ds.X_train[:, 17] + \
    #         #  beta[18] * ds.X_train[:, 18] + beta[19] * ds.X_train[:, 19] + \
    #         #  beta[20] * ds.X_train[:, 20] + beta[21] * ds.X_train[:, 21] + \
    #         #  beta[22] * ds.X_train[:, 22] + beta[23] * ds.X_train[:, 23] + \
    #         #  beta[24] * ds.X_train[:, 24] + beta[25] * ds.X_train[:, 25] + \
    #         #  beta[26] * ds.X_train[:, 26] + beta[27] * ds.X_train[:, 27] + \
    #         #  beta[28] * ds.X_train[:, 28] + beta[29] * ds.X_train[:, 29]

    # y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=ds.Y_train[:, 1])

    # n_sample = 500
    # with model:
    #     # start = pm.find_MAP()
    #     step = pm.NUTS()
    #     trace = pm.sample(n_sample, tune=2000)

    # # pm.traceplot(trace)
    # pm.plot_posterior(trace, figsize=(14, 14))
    # plt.show()

    # summary = pm.summary(trace).round(4)
    # print(summary)
    # # pm.energyplot(trace)

    # # with model:
    # #     post_pred = pm.sample_posterior_predictive(trace, samples=500)
