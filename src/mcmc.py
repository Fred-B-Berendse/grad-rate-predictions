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

    def get_prediction(self, X_arr):
        y_pred = []
        for target_label in self.dataset.target_labels:
            trace = self.traces[target_label]
            n_traces = len(trace) // 2
            parameters = pm.summary(trace[-n_traces:]).values
            intercept = parameters[0, 0]
            coeffs = parameters[1:-1, 0]
            y_pred.append(intercept + np.dot(X_arr, coeffs))
        return np.array(y_pred).T

    def predict(self, samples=500, size=50):
        self.train_predict = self.get_prediction(self.dataset.X_train)
        self.test_predict = self.get_prediction(self.dataset.X_test)
        self.train_residuals = self.dataset.Y_train - self.train_predict
        self.test_residuals = self.dataset.Y_test - self.test_predict

    def plot_coeff_distribution(self, target_label):
        n_features = len(self.dataset.feature_labels)
        n_traces = len(self.traces[target_label]) // 2
        trace = self.traces[target_label][-n_traces:]
        labels = trace.varnames[:-2]
        pos = list(range(-1, -n_features-2, -1))

        trace_vals = []
        for la in labels:
            trace_vals.append(trace[la])

        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(111)
        parts = ax.violinplot(trace_vals, pos, points=80, vert=False,
                              widths=0.7, showmeans=True, showextrema=True,
                              showmedians=False)

        target_color = self.dataset.target_colors[target_label]
        for pc in parts['bodies']:
            pc.set_facecolor(target_color)
            pc.set_color(target_color)
            pc.set_edgecolor('black')
        parts['cmeans'].set_color(target_color)
        parts['cbars'].set_color(target_color)
        parts['cmins'].set_color(target_color)
        parts['cmaxes'].set_color(target_color)

        ax.set_title('Coefficients: ' + target_label + ' Graduation Rate')
        ax.set_xlabel('Coefficient')
        ax.set_yticks(pos)
        ax.set_yticklabels([la for la in labels])

        ax.axvline(x=0, color='blue', linestyle='--')
        plt.tight_layout()
        return fig, ax

    def plot_coeff_distributions(self):
        for target_label in self.dataset.target_labels:
            self.plot_coeff_distribution(target_label)

    def plot_rate_distribution(self, target_label, ax,
                               pos=0, samples=500, size=50):
        trace = self.traces[target_label]
        n_traces = len(trace) // 2
        model = self.models[target_label]
        ppc = pm.sample_ppc(trace[-n_traces:], samples=samples,
                            model=model, size=size)
        y_pred_dist = ppc['y'].mean(1).mean(1)
        parts = ax.violinplot(y_pred_dist, [pos], points=80, vert=False,
                              widths=0.7, showmeans=True, showextrema=True,
                              showmedians=False)

        # Plot the actual mean graduation rate
        idx = np.argwhere(self.dataset.target_labels == target_label)[0][0]
        ax.scatter(self.dataset.Y_train[:, idx].mean(), pos,
                   marker='D', label='Actual', color='black')

        target_color = self.dataset.target_colors[target_label]
        for pc in parts['bodies']:
            pc.set_facecolor(target_color)
            pc.set_color(target_color)
            pc.set_edgecolor('black')
        parts['cmeans'].set_color(target_color)
        parts['cbars'].set_color(target_color)
        parts['cmins'].set_color(target_color)
        parts['cmaxes'].set_color(target_color)

    def plot_rate_distributions(self, samples=500, size=50):
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(111)
        n_targets = self.dataset.n_targets
        pos = range(-1, -n_targets-1, -1)
        for target_label, p in zip(self.dataset.target_labels, pos):
            self.plot_rate_distribution(target_label, ax, pos=p,
                                        samples=samples, size=size)

        ax.set_title('Predicted Graduation Rates')
        ax.set_xlabel('Graduation Rate')
        ax.set_yticks(pos)
        ax.set_yticklabels([la for la in self.dataset.target_labels])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], [labels[0]], loc='lower right')


if __name__ == "__main__":

    build_model = True

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')

    feat_cols = np.array(['en25', 'upgrntp', 'latitude', 'control_privnp',
                          'locale_twnrem', 'locale_rurrem', 'grntof2_pct',
                          'uagrntp', 'enrlt_pct'])

    target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
                            'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
                            'cstcball_pct_grwhitt', 'pgcmbac_pct',
                            'sscmbac_pct', 'nrcmbac_pct'])
    # target_cols = np.array(['cstcball_pct_gr2mort', 'cstcball_pct_grasiat'])

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.target_labels = np.array(['2+ Races', 'Asian', 'Black', 'Hispanic',
                                 'White', 'Pell Grant', 'SSL',
                                 'Non-Recipient'])
    # ds.target_labels = np.array(['2+ Races', 'Asian'])
    ds.target_colors = targets_color_dict()

    tr_feature_dict = {'grntof2_pct': ('log_grntof2_pct', ds.log10_sm),
                       'uagrntp': ('logu_uagrntp', ds.log10u_sm),
                       'enrlt_pct': ('log_enrlt_pct', ds.log10_sm)}
    ds.transform_features(tr_feature_dict, drop_old=True)

    mcmc = McmcRegressor(ds)

    filepath = 'data/mcmc.pkl'
    if build_model:
        mcmc.build_models(draws=2000, tune=500)
        mcmc.pickle_model(filepath)
    else:
        with open(filepath, 'rb') as buff:
            data = pickle.load(buff)
        mcmc.models, mcmc.traces = data['models'], data['traces']

    # Predict test and train data
    mcmc.predict()
    train_r2, test_r2 = mcmc.r_squared()
    train_rmse, test_rmse = mcmc.rmse(unscale=False)
    for i, f in enumerate(ds.target_labels):
        print("{}".format(f))
        formstr = "  train R^2: {:.3f}; test R^2: {:.3f}"
        print(formstr.format(train_r2[i], test_r2[i]))
        formstr = "  train RMSE: {:.2f}; test RMSE: {:.2f}"
        print(formstr.format(train_rmse[i], test_rmse[i]))

    # Generate distribution of coefficients for each target
    mcmc.plot_coeff_distributions()
    plt.show()

    # Generate distributions of graduation rates for each target
    mcmc.plot_rate_distributions()
    plt.show()
