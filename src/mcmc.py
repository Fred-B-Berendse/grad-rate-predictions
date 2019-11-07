import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from colors import targets_color_dict
from dataset import Dataset
from regressor import Regressor
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


class McmcRegressor(Regressor):

    def __init__(self, dataset):
        self.model = pm.Model()
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
    def replace_bad_chars(mystr):
        res = mystr.replace(' ', '_').replace('+', 'pl')
        res = res.replace('-', '_').replace('2', 'two')
        return res.lower()

    def _make_formula(self, feat_list, target_label):
        tl = self.replace_bad_chars(target_label)
        return tl + ' ~ ' + ' + '.join(feat_list)

    def _format_data(self, target_label):
        data = pd.DataFrame(data=self.dataset.X_train,
                            columns=self.dataset.feature_labels)
        tar_col = self.replace_bad_chars(target_label)
        idx = np.argwhere(self.dataset.target_labels == target_label)[0][0]
        data[tar_col] = self.dataset.Y_train[:, idx]
        return data

    def build_model(self, target_label, draws=2000, tune=500):

        data = self._format_data(target_label)
        print("data columns: {}".format(data.columns))
        formula = self._make_formula(self.dataset.feature_labels,
                                     target_label)
        print("formula: {}".format(formula))

        with self.model:
            # family = pm.glm.families.Normal()
            pm.glm.GLM.from_formula(formula, data)

        with self.model:
            self.trace = pm.sample(draws, tune=tune)

    def plot_posterior(self, **kwargs):
        pm.plot_posterior(self.trace, **kwargs)

    def pickle_model(self, filepath):
        with open(filepath, 'wb') as buff:
            pickle.dump({'model': self.model, 'trace': self.trace}, buff)


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')

    feat_cols = np.array(['en25', 'upgrntp', 'latitude', 'control_privnp',
                          'locale_twnrem', 'locale_rurrem', 'grntof2_pct',
                          'uagrntp', 'enrlt_pct'])

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

    tr_feature_dict = {'grntof2_pct': ('log_grntof2_pct', ds.log10_sm),
                       'uagrntp': ('logu_uagrntp', ds.log10u_sm),
                       'enrlt_pct': ('log_enrlt_pct', ds.log10_sm)}
    ds.transform_features(tr_feature_dict, drop_old=True)

    # for target_label in ds.target_labels:
    #     mcmc = McmcRegressor(ds)
    #     mcmc.build_model(target_label)
    #     filepath = 'data/mcmc_' + mcmc.replace_bad_chars(target_label) + '.pkl'
    #     print(filepath)
    #     mcmc.pickle_model(filepath)

    for target_label in ds.target_labels:
        mcmc = McmcRegressor(ds)
        filepath = 'data/mcmc_' + mcmc.replace_bad_chars(target_label) + '.pkl'
        with open(filepath, 'rb') as buff:
            data = pickle.load(buff)
        mcmc.model, mcmc.trace = data['model'], data['trace']

        summary = pm.summary(mcmc.trace).round(4)
        print("Summary for {}".format(target_label))
        print(summary)

        pm.plot_posterior(mcmc.trace, figsize=(14, 14))

    plt.show()
