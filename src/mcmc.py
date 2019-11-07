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
        self.dataset = dataset
        target_labels = self.dataset.target_labels
        self.models = dict(zip(target_labels,
                               [pm.Model() for _ in target_labels]))
        self.traces = dict(zip(target_labels,
                               [None for _ in target_labels]))
        self.var_means = dict(zip(target_labels,
                                  [None for _ in target_labels]))
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

        with self.models[target_label]:
            # family = pm.glm.families.Normal()
            pm.glm.GLM.from_formula(formula, data)

        with self.models[target_label]:
            self.traces[target_label] = pm.sample(draws, tune=tune)

    def build_models(self, draws=2000, tune=500):
        for i, target_label in enumerate(self.dataset.target_labels):
            self.build_model(target_label, draws=draws, tune=tune)

    def plot_posterior(self, **kwargs):
        pm.plot_posterior(self.trace, **kwargs)

    def pickle_model(self, filepath):
        with open(filepath, 'wb') as buff:
            pickle.dump({'models': self.models, 'traces': self.traces}, buff)

    def calc_var_means(self):
        self.var_means = {}
        for variable in self.trace.varnames:
            mean = np.mean(self.trace[variable])
            self.var_means[variable] = mean

    def predict_one(self, X, get_range=False):

        # Insert Intercept to labels and X values
        X_obs = np.insert(X, 0, 1)
        labels = np.insert(self.dataset.feature_labels, 0, 'Intercept')

        # Align weights with labels
        var_means = pd.DataFrame(self.var_means, index=[0])
        var_means = var_means[labels]

        # Calculate the mean for observation
        mean_loc = np.dot(var_means, X_obs)[0]

        if get_range:
            sd_value = self.var_means['sd']
            estimates = np.random.normal(loc=mean_loc, scale=sd_value,
                                         size=1000)
            hpd_lo = np.percentile(estimates, 2.5)
            hpd_hi = np.percentile(estimates, 97.5)
            return mean_loc, hpd_lo, hpd_hi
        else:
            return mean_loc

    def predict_target(self, target_label, samples=500, size=50):
        trace = self.traces[target_label]
        model = self.models[target_label]
        n_traces = len(trace) // 2
        ppc = pm.sample_ppc(trace[-n_traces:], samples=samples,
                            model=model, size=size)
        return ppc['y'].mean(0).mean(0).T

    def predict(self):
        pass
        # self.train_predict = self.model.predict(self.dataset.X_train)
        # self.train_residuals = self.dataset.Y_train - self.train_predict
        # self.test_predict = self.model.predict(self.dataset.X_test)
        # self.test_residuals = self.dataset.Y_test - self.test_predict


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')

    feat_cols = np.array(['en25', 'upgrntp', 'latitude', 'control_privnp',
                          'locale_twnrem', 'locale_rurrem', 'grntof2_pct',
                          'uagrntp', 'enrlt_pct'])

    # target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
    #                         'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
    #                         'cstcball_pct_grwhitt', 'pgcmbac_pct',
    #                         'sscmbac_pct', 'nrcmbac_pct'])
    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat'])

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    # ds.target_labels = np.array(['2+ Races', 'Asian', 'Black', 'Hispanic',
    #                              'White', 'Pell Grant', 'SSL',
    #                              'Non-Recipient'])
    ds.target_labels = np.array(['2+ Races', 'Asian'])
    ds.target_colors = targets_color_dict()

    tr_feature_dict = {'grntof2_pct': ('log_grntof2_pct', ds.log10_sm),
                       'uagrntp': ('logu_uagrntp', ds.log10u_sm),
                       'enrlt_pct': ('log_enrlt_pct', ds.log10_sm)}
    ds.transform_features(tr_feature_dict, drop_old=True)

    mcmc = McmcRegressor(ds)

    # mcmc.build_models(draws=200, tune=100)
    # filepath = 'data/mcmc.pkl'
    # mcmc.pickle_model(filepath)

    filepath = 'data/mcmc.pkl'
    with open(filepath, 'rb') as buff:
        data = pickle.load(buff)
    mcmc.models, mcmc.traces = data['models'], data['traces']

    y_pred = mcmc.predict_target('Asian', samples=100, size=50)

    # for j, target_label in enumerate(ds.target_labels):
    #     mcmc = McmcRegressor(ds)
    #     filepath = 'data/mcmc_' + mcmc.replace_bad_chars(target_label) + '.pkl'
    #     with open(filepath, 'rb') as buff:
    #         data = pickle.load(buff)
    #     mcmc.model, mcmc.trace = data['model'], data['trace']

    #     summary = pm.summary(mcmc.trace).round(4)
    #     print("Summary for {}".format(target_label))
    #     print(summary)

    #     mcmc.calc_var_means()
    #     print("Prediction for {}".format(target_label))
    #     test_obs = mcmc.dataset.X_test[0, :]
    #     mn, lo, hi = mcmc.predict_one(test_obs, get_range=True)
    #     print("  Predicted rate: {:.0f} ({:.0f} - {:.0f})".format(mn, lo, hi))
    #     print("  Actual: {:.0f}".format(mcmc.dataset.Y_test[0, j]))
    #     Y_pred = mcmc.predict_all(mcmc.dataset.X_test)
    #     pm.plot_posterior(mcmc.trace, figsize=(14, 14))

    plt.show()
