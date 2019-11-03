import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from plotting import make_barplot, plot_partial_dependence
from regressor import Regressor
from copy import deepcopy
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


def get_scores(model, X_train, Y_train, X_test, Y_test):
    train_scores = []
    test_scores = []
    for i in range(Y_train.shape[1]):
        model.fit(X_train, Y_train[:, i])
        train_scores.append(model.score(X_train, Y_train[:, i]))
        test_scores.append(model.score(X_test, Y_test[:, i]))
    return train_scores, test_scores


def print_metrics(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_resid = Y_test - Y_pred
    avg_abs_err = np.mean(np.abs(Y_resid), axis=0)
    rmse = np.sqrt(np.mean(Y_resid**2, axis=0))
    train_r2, test_r2 = get_scores(deepcopy(model), X_train, Y_train, X_test,
                                   Y_test)
    print(f"  R^2 train: {train_r2}")
    print(f"  R^2 test: {test_r2}")
    print(f"  Avg absolute error: {avg_abs_err}")
    print(f"  RMSE: {rmse}")
    print(f"  Parameters: {model.get_params()}")
    return


class ForestRegressor(Regressor):

    def __init__(self):
        self.feature_importances = np.array([])
        self.partial_dependences = np.array([])

    def calc_feature_importances(self):
        pass

    def plot_feature_importances(self, n_features=None, targets_list=None):
        pass

    def calc_partial_dependences(self):
        pass

    def plot_partial_dependences(self, features_list, targets_list):
        pass


if __name__ == "__main__":

    mdf = pd.read_csv('data/ipeds_2017_model.csv')
    mdf.drop(['Unnamed: 0', 'applcn'], axis=1, inplace=True)

    feat_cols = np.array(['longitud', 'latitude', 'enrlft_pct',
                          'enrlt_pct', 'admssn_pct', 'en25',
                          'uagrntp', 'upgrntp', 'grntwf2_pct',
                          'grntof2_pct'])

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

    base_reg = RandomForestRegressor(n_estimators=100, criterion='mse',
                                     min_samples_split=2, min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_features='sqrt', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=-1,
                                     random_state=10, warm_start=False)

    # Get metrics for a baseline model
    print("Baseline Model")
    print_metrics(base_reg, X_train, Y_train, X_test, Y_test)

    # Use random search CV to get best hyperparameters
    grid = {'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
            'criterion': ['mse', 'mae'],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['auto', 'sqrt']}

    rf_random = RandomizedSearchCV(base_reg, grid, n_iter=25, scoring=None,
                                   n_jobs=-1, iid=False,
                                   refit=True, cv=5, verbose=2,
                                   pre_dispatch='2*n_jobs', random_state=10,
                                   error_score=np.nan,
                                   return_train_score=False)
    rf_random.fit(X_train, Y_train)
    print(rf_random.best_params_)
    best_random = rf_random.best_estimator_

    best_rf = RandomForestRegressor(n_estimators=600, criterion='mae',
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    min_weight_fraction_leaf=0.0,
                                    max_features='sqrt',
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0.0,
                                    bootstrap=True,
                                    oob_score=False, n_jobs=-1,
                                    random_state=10, warm_start=False)

    print("Best Random Model")
    print_metrics(best_rf, X_train, Y_train, X_test, Y_test)

    # Feature Importance plots
    for j in range(len(target_cols)):
        print(f"Target: {target_cols[j]}")
        best_rf.fit(X_train, Y_train[:, j])
        fig, ax = make_barplot(best_rf.feature_importances_,
                               x_labels=feat_cols, color=colors[j],
                               y_label='Feature Importance')
        ax.tick_params(axis='x', rotation=70)
        ax.set_title(f'{labels[j]} Graduation Rate')
        plt.show()

    # Plot all racial partial dependences on one plot
    top_features = [5, 7, 4, 8]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for j in range(len(target_cols[:-3])):
        best_rf.fit(X_train, Y_train[:, j])
        for i, f in enumerate(top_features):
            axi = ax[i//2, i % 2]
            plot_partial_dependence(best_rf, X_test, f, axi, color=colors[j],
                                    label=labels[j])

    for i, f in enumerate(top_features):
        axi = ax[i//2, i % 2]
        axi.set_xlabel(feat_cols[f])
        axi.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot all Pell Grant/SSL status partial dependences on one plot
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    offset = len(race_labels)
    for j in range(len(target_cols[-3:])):
        best_rf.fit(X_train, Y_train[:, j+offset])
        for i, f in enumerate(top_features):
            axi = ax[i//2, i % 2]
            plot_partial_dependence(best_rf, X_test, f, axi,
                                    color=colors[j+offset],
                                    label=labels[j+offset])

    for i, f in enumerate(top_features):
        axi = ax[i//2, i % 2]
        axi.set_xlabel(feat_cols[f])
        axi.legend(loc='best')
    plt.tight_layout()
    plt.show()