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
    beta = pm.Normal('beta', 0, 10, shape=2)
    sigma = pm.HalfNormal('sigma', 10)
    mu = alpha + beta[0]*ds.X_train[:, 25] + beta[1]*ds.X_train[:, 24]
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

# fig, ax = plt.subplots()
# ax.scatter(df.mpg, df.horsepower, s=10)
# ax.set_xlabel('mpg')
# ax.set_ylabel('horsepower')
# ax.set_title('Cars, with Distribution of Possible Linear Fits')
# xlim = np.array(ax.get_xlim())
# for i in range(0, n_sample, 10):
#     ax.plot(xlim, trace[i]['beta0'] + trace[i]['beta1'] * xlim,
#             c='k', lw=1, alpha=0.1)
# ax.set_ylim(bottom=0)
# ax.set_xlim(xlim)