import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from plotting import make_barplot, plot_partial_dependence
from regressor import Regressor
from copy import deepcopy
from colors import targets_color_dict
from dataset import Dataset
from joblib import dump, load
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


class ForestRegressor(Regressor):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.train_predict = None
        self.test_predict = None
        self.train_residuals = None
        self.test_residuals = None
        self.feature_importances = None
        self.partial_dependences = None

    def plot_feature_importances(self, n_features=None):
        model = deepcopy(self.model)
        for j, target in enumerate(self.dataset.target_labels):
            model.fit(self.dataset.X_train, self.dataset.Y_train[:, j])
            idx = np.argsort(model.feature_importances_)
            idx = idx[-1:-n_features-1:-1]
            fig, ax = make_barplot(model.feature_importances_[idx],
                                   x_labels=self.dataset.feature_labels[idx],
                                   y_label='Feature Importance',
                                   color=self.dataset.target_colors[target])
            ax.tick_params(axis='x', rotation=70)
            ax.set_title(target)

    def plot_partial_dependences(self, features_list, targets_list, 
                                 desc_dict=None):
        model = deepcopy(self.model)
        feat_idx = [np.where(self.dataset.feature_labels == f)
                    for f in features_list]
        feat_idx = np.array(feat_idx).flatten()
        tar_idx = [np.where(self.dataset.target_labels == t)
                   for t in targets_list]
        tar_idx = np.array(tar_idx).flatten()
        n_features = len(feat_idx)
        nrows = (n_features - 1) // 2 + 1
        fig, ax = plt.subplots(nrows, 2, figsize=(12, 6*nrows))
        for j in tar_idx:
            model.fit(self.dataset.X_train, self.dataset.Y_train[:, j])
            for i, fi in enumerate(feat_idx):
                axi = ax[i // 2, i % 2]
                tar_label = self.dataset.target_labels[j]
                color = self.dataset.target_colors[tar_label]
                plot_partial_dependence(model, self.dataset.X_test, fi, axi,
                                        color=color, label=tar_label)

        for i, f in enumerate(features_list):
            axi = ax[i // 2, i % 2]
            if desc_dict is not None:
                axi.set_xlabel(desc_dict[f])
            else:
                axi.set_xlabel(f)
            axi.legend(loc='best')
        plt.tight_layout()

    def print_metrics(self):
        mae = self.mae()
        rmse = self.rmse()
        r2 = self.r_squared()
        print("Parameters:")
        print(self.model.get_params())
        print("\n")
        for i, tar in enumerate(self.dataset.target_labels):
            print("Target: {}".format(tar))
            formstr = "   R^2 train: {:.3f}; test: {:.3f}"
            print(formstr.format(r2[0][i], r2[1][i]))
            formstr = "   MAE train: {:.3f}; test: {:.3f}"
            print(formstr.format(mae[0][i], mae[1][i]))
            formstr = "   RMSE train: {:.3f}; test: {:.3f}"
            print(formstr.format(rmse[0][i], rmse[1][i]))


if __name__ == "__main__":

    do_grid_search = False

    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')
    mdf.drop(['Unnamed: 0', 'applcn'], axis=1, inplace=True)

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
    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0.25,
                         random_state=10)
    ds.target_labels = np.array(['2+ Races', 'Asian', 'Black', 'Hispanic',
                                 'White', 'Pell Grant', 'SSL',
                                 'Non-Recipient'])
    ds.target_colors = targets_color_dict()

    # Fit a baseline model first
    base_model = RandomForestRegressor(n_estimators=100, criterion='mse',
                                       min_samples_split=2, min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_features='auto',
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       bootstrap=True,
                                       oob_score=False, n_jobs=-1,
                                       random_state=10, warm_start=False)
    rfbase = ForestRegressor(base_model, ds)
    rfbase.fit_train()
    rfbase.predict()
    print("Baseline Model")
    rfbase.print_metrics()
    print("\n")

    if do_grid_search:

        # Use Grid Search to find best hyperparameters
        grid = {'n_estimators': [int(x) for x in np.linspace(100, 200, 11)],
                'criterion': ['mse', 'mae'],
                'min_samples_split': [int(x) for x in np.linspace(2, 10, 2)],
                'max_features': ['auto', 'sqrt']}

        search_model = GridSearchCV(base_model, grid,
                                    scoring=None, n_jobs=-1, iid=False,
                                    refit=True, cv=5, verbose=2,
                                    pre_dispatch='2*n_jobs',
                                    error_score=np.nan,
                                    return_train_score=False)
        rfsearch = ForestRegressor(search_model, ds)
        rfsearch.fit_train()
        print("Best Hyperparameters: ")
        print(rfsearch.model.best_params_)
        best_model = rfsearch.model.best_estimator_
        # Use joblib instead of pickle to save the model
        dump(best_model, 'data/forest.joblib')
    else:
        best_model = load('data/forest.joblib')

    rfbest = ForestRegressor(best_model, ds)
    rfbest.fit_train()
    rfbest.predict()
    print("Best Model")
    rfbest.print_metrics()
    print("\n")

    rfbest.plot_feature_importances(n_features=8)
    plt.show()

    desc_dict = {'en25': 'English 25th Percentile', 
                 'upgrntp': 'Percent Receiving Pell Grant',
                 'admssn_pct': 'Percent of Applicants Admitted',
                 'grntwf2_pct': 'Percent Living with Family'}

    top_features = ['en25', 'upgrntp', 'admssn_pct', 'grntwf2_pct']
    targets = ['2+ Races', 'Asian', 'Black', 'Hispanic', 'White']
    rfbest.plot_partial_dependences(top_features, targets, desc_dict=desc_dict)
    plt.show()

    targets = ['Pell Grant', 'SSL', 'Non-Recipient']
    rfbest.plot_partial_dependences(top_features, targets, desc_dict=desc_dict)
    plt.show()
