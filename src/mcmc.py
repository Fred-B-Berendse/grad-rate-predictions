import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import Dataset
from colors import targets_color_dict
import pymc3 as pm
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


if __name__ == "__main__":

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
                          'en25', 'uagrntp', 'upgrntp',
                          'npgrn2', 'grntof2_pct', 'grntwf2_pct'])

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


with pm.Model() as model:
    alpha = pm.Normal('alpha', 0, 20)
    beta = pm.Normal('beta', 0, 10, shape=30)
    sigma = pm.HalfNormal('sigma', 10)
    mu = alpha + beta*ds.X_train
    # mu = alpha + beta[0]*ds.X_train[:, 0] + beta[1]*ds.X_train[:, 1] + \
    #      beta[2] * ds.X_train[:, 2] + beta[3] * ds.X_train[:, 3] + \
    #      beta[4] * ds.X_train[:, 4] + beta[5] * ds.X_train[:, 5] + \
    #      beta[6] * ds.X_train[:, 6] + beta[7] * ds.X_train[:, 7] + \
    #      beta[8] * ds.X_train[:, 8] + beta[9] * ds.X_train[:, 9] + \
    #      beta[10] * ds.X_train[:, 10] + beta[11] * ds.X_train[:, 11] + \
    #      beta[12] * ds.X_train[:, 12] + beta[13] * ds.X_train[:, 13] + \
    #      beta[14] * ds.X_train[:, 14] + beta[15] * ds.X_train[:, 15] + \
    #      beta[16] * ds.X_train[:, 16] + beta[17] * ds.X_train[:, 17] + \
    #      beta[18] * ds.X_train[:, 18] + beta[19] * ds.X_train[:, 19] + \
    #      beta[20] * ds.X_train[:, 20] + beta[21] * ds.X_train[:, 21] + \
    #      beta[22] * ds.X_train[:, 22] + beta[23] * ds.X_train[:, 23] + \
    #      beta[24] * ds.X_train[:, 24] + beta[25] * ds.X_train[:, 25] + \
    #      beta[26] * ds.X_train[:, 26] + beta[27] * ds.X_train[:, 27] + \
    #      beta[28] * ds.X_train[:, 28] + beta[29] * ds.X_train[:, 29] + \
    #      beta[30] * ds.X_train[:, 30]

    y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=ds.Y_train[:, 0])

n_sample = 500
with model:
    # start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(n_sample, tune=2000)

pm.traceplot(trace)

pm.summary(trace).round(2)
pm.gelman_rubin(trace)
pm.energyplot(trace)
plt.show()

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
