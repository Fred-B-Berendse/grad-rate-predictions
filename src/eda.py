import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from plotting import make_histograms, make_rank_plot
from plotting import make_barplot, make_stacked_barplot, make_violin_plots
from plotting import scree_plot, make_embedding_graph
from colors import targets_color_dict, get_colors
from dataset import Dataset
from sklearn.decomposition import PCA
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


class PCAModel(object):

    def __init__(self, dataset, n_components=5):
        self.model = PCA(n_components=n_components)
        self.dataset = dataset
        self.n_components = n_components
        if self.dataset.features_scaler is None:
            self.dataset.scale_features()
        self.model.fit(self.dataset.X_train)
        self.components = self.model.components_
        self.explained_variance = self.model.explained_variance_

    def plot_embedding(self, n_obs=None, n_digits=0):
        '''
        plots an embedding of the targets onto the first two principal
        components

        n_obs - if None, all training observations are graphed,
                otherwise only n_obs random observations are graphed
        '''
        for i, l in enumerate(self.dataset.target_labels):
            y_rounded = self.dataset.Y_train[:, i].round(n_digits)
            fig, ax = make_embedding_graph(self.dataset.X_train, y_rounded,
                                           l, n=100)
            ax.set_title(f'PCA: {l}')
            ax.set_xlabel('First principal component axis')
            ax.set_ylabel('Second principal component axis')

    def plot_variance(self, n_components=None):
        '''
        plots the variance explained by each of the principal components
        as a line graph
        '''
        if n_components is None:
            n_components = self.n_components

        fig, ax = scree_plot(self.model, n_components_to_plot=n_components)
        ax.set_title("Variance Plot for Principal Components")
        ax.set_xlabel('Principal Component Axis Number')
        ax.set_ylabel('Percent of Total Variance')


if __name__ == "__main__":

    # Read in the dataframe without one-hot-encoded categories
    mdf = pd.read_csv('data/ipeds_2017_eda.csv')
    mdf.drop('Unnamed: 0', axis=1, inplace=True)

    # Make color dictionaries
    tar_color_dict = targets_color_dict()

    # Make barplots of select categorical values
    select_cols = ['control', 'hloffer', 'hbcu', 'locale',
                   'landgrnt', 'instsize']
    colors = cm.Dark2(np.linspace(0, 1, len(select_cols)))
    for i, col in enumerate(select_cols):
        val_cnts = mdf[col].value_counts()
        labels = val_cnts.index.values
        vals = val_cnts.values
        fig, ax = make_barplot(vals, x_labels=labels, width=0.8)
        ax.set_title(col)
        ax.set_xlabel('Category')
        ax.tick_params(axis='x', rotation=70)
    plt.show()

    # Make histograms of select features
    group = ['longitud', 'latitude',
             'admssn_pct', 'enrlt_pct',
             'enrlft_pct', 'npgrn2',
             'en25', 'en75',
             'mt25', 'mt75',
             'uagrntp', 'upgrntp',
             'grnton2_pct', 'grntwf2_pct',
             'grntof2_pct']
    labels = ['Longitude', 'Latitude',
              'Percent Admitted', 'Percent Enrolled',
              'Percent Full Time', 'Net Price: Pell Grant Students',
              'Scaled English 25th Percentile',
              'Scaled English 75th Percentile',
              'Scaled Math 25th Percentile',
              'Scaled Math 75th Percentile',
              'Percent Awarded Aid (any)',
              'Percent Awarded Pell Grant',
              'Percent On Campus: 2016-17',
              'Percent With Family: 2016-17',
              'Percent Off Campus: 2016-17']
    make_histograms(mdf.loc[:, group].values, x_labels=labels)
    plt.show()

    # Make histograms of graduation rates
    group = ['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
             'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
             'cstcball_pct_grwhitt']
    labels = ['2+ Races', 'Asian', 'Black', 'Hispanic', 'White']
    x_labels = ['Graduation Rate: ' + l for l in labels]
    colors = [tar_color_dict[l] for l in labels]
    make_histograms(mdf.loc[:, group].values, x_labels=x_labels, colors=colors)
    plt.show()

    group = ['pgcmbac_pct', 'sscmbac_pct', 'nrcmbac_pct']
    labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    x_labels = ['Graduation Rate: ' + l for l in labels]
    colors = [tar_color_dict[l] for l in labels]
    make_histograms(mdf.loc[:, group].values, x_labels=x_labels, colors=colors)
    plt.show()

    # Make histograms of graduation ratios
    group = ['cstcball_rat_gr2mort', 'cstcball_rat_grasiat',
             'cstcball_rat_grbkaat', 'cstcball_rat_grhispt']
    labels = ['2+ Races', 'Asian', 'Black', 'Hispanic']
    x_labels = ['Graduation Rate Ratio: ' + l + ' to White' for l in labels]
    colors = [tar_color_dict[l] for l in labels]
    make_histograms(mdf.loc[:, group].values, x_labels=x_labels, colors=colors)
    plt.show()

    group = ['pgcmbac_rat', 'sscmbac_rat']
    labels = ['Pell Grant', 'SSL']
    x_labels = ['Graduation Rate: ' + l + ' to Non-Recipient' for l in labels]
    colors = [tar_color_dict[l] for l in labels]
    make_histograms(mdf.loc[:, group].values, x_labels=x_labels, colors=colors)
    plt.show()

    # institutional rank percentiles vs SAT/ACT benchmark quartiles
    fig, ax = make_rank_plot(mdf, ['satvr25', 'acten25'], 'en25',
                             ['SAT Verbal 25th Percentile',
                             'ACT English 25th Percentile'])

    fig, ax = make_rank_plot(mdf, ['satvr75', 'acten75'], 'en75',
                             ['SAT Verbal 75th Percentile',
                             'ACT English 75th Percentile'])

    fig, ax = make_rank_plot(mdf, ['satmt25', 'actmt25'], 'mt25',
                             ['SAT Math 25th Percentile',
                             'ACT Math 25th Percentile'])

    fig, ax = make_rank_plot(mdf, ['satmt75', 'actmt75'], 'mt75',
                             ['SAT Math 75th Percentile',
                             'ACT Math 75th Percentile'])
    plt.show()

    # Sum of all students
    columns = ['cstrevex_grwhitt', 'cstrevex_grhispt', 'cstrevex_grbkaat',
               'cstrevex_grasiat', 'cstrevex_graiant', 'cstrevex_grnhpit',
               'cstrevex_gr2mort']
    print("Total cohort count: ", mdf.loc[:, columns].sum().sum())

    # Completion counts & US Population
    columns = ['cstcball_grwhitt', 'cstcball_grhispt', 'cstcball_grbkaat',
               'cstcball_grasiat', 'cstcball_graiant', 'cstcball_grnhpit', 
               'cstcball_gr2mort']
    labels = ['White', 'Hispanic', 'Black', 'Asian', 'Nat. Am.', 'Pac. Isl.',
              '2+ Races']
    colors = get_colors(labels, tar_color_dict)
    completions = mdf.loc[:, columns].sum(axis=0).values
    sum_completions = completions.sum()
    print("Total completions: ", sum_completions)
    census_pct = np.array([60.4, 18.3, 13.4, 5.9, 1.3, 0.2, 2.7])
    census_ct = census_pct / census_pct.sum() * sum_completions
    x_labels = ["Completions", "US Census"]
    y_label = 'Number of students'
    bars = [completions, census_ct]
    fig, ax = make_stacked_barplot(bars, x_labels=x_labels, y_label=y_label,
                                   colors=colors, stack_labels=labels, width=3)
    plt.show()

    # Violin plots of completion percentages
    columns = ['pgcmbac_pct', 'sscmbac_pct', 'nrcmbac_pct']
    col_labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    colors = get_colors(col_labels, tar_color_dict)
    fig, ax, colors_ps = make_violin_plots(mdf, columns, col_labels,
                                           colors=colors)
    ax.set_ylabel('Percentage of Completions')
    ax.set_title("Institution-wide Completion Percentage by Recipient Status")
    plt.show()

    columns = ['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
               'cstcball_pct_grbkaat', 'cstcball_pct_grhispt',
               'cstcball_pct_grwhitt']
    col_labels = ['2+ Races', 'Asian', 'Black', 'Hispanic', 'White']
    colors = get_colors(col_labels, tar_color_dict)
    fig, ax, colors_race = make_violin_plots(mdf, columns, col_labels,
                                             colors=colors)
    ax.set_ylabel('Percentage of Completions')
    ax.set_title("Institution-wide Completion Percentage by Race/Ethnicity")
    plt.show()

    # PCA plots: completion percentages
    feat_cols = ['iclevel_2to4', 'iclevel_0to2', 'iclevel_na',
                 'control_public', 'control_privnp', 'control_na',
                 'hloffer_assoc', 'hloffer_doct', 'hloffer_bach',
                 'hloffer_mast', 'hloffer_2to4yr', 'hloffer_0to1yr',
                 'hloffer_postmc', 'hloffer_na', 'hloffer_postbc',
                 'hbcu_yes', 'locale_ctylrg', 'locale_ctysml',
                 'locale_ctymid', 'locale_twndst', 'locale_rurfrg',
                 'locale_twnrem', 'locale_submid', 'locale_subsml',
                 'locale_twnfrg', 'locale_rurdst', 'locale_rurrem',
                 'locale_na', 'instsize_1to5k', 'instsize_5to10k',
                 'instsize_10to20k', 'instsize_na', 'instsize_gt20k',
                 'instsize_norpt', 'landgrnt_yes', 'longitud', 'latitude',
                 'admssn_pct', 'enrlt_pct', 'enrlft_pct', 'en25', 'en75',
                 'mt25', 'mt75', 'uagrntp', 'upgrntp', 'npgrn2', 'grnton2_pct']

    target_cols = ['cstcball_pct_grwhitt', 'cstcball_pct_grbkaat',
                   'cstcball_pct_grhispt', 'cstcball_pct_grasiat',
                   'cstcball_pct_gr2mort', 'pgcmbac_pct', 'sscmbac_pct',
                   'nrcmbac_pct']

    # Read in the dataframe with one-hot-encoded categories
    mdf = pd.read_csv('data/ipeds_2017_cats_eda.csv')
    mdf.drop('Unnamed: 0', axis=1, inplace=True)
    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0,
                         random_state=10)

    labels = ['White', 'Black', 'Hispanic', 'Asian', '2+ Races', 'Pell Grant',
              'SSL', 'Non-Recipient']
    ds.target_labels = [l + ' Graduation Rate' for l in labels]
    race_colors = get_colors(labels[:-3], tar_color_dict)
    ps_colors = get_colors(labels[-3:], tar_color_dict)
    ds.target_colors = np.append(race_colors, ps_colors, axis=0)
    pca_pct = PCAModel(ds)
    pca_pct.plot_embedding(n_obs=100)
    plt.show()

    idx = np.argsort(abs(pca_pct.components[0]))[:-11:-1]
    print("Rates: Heavy features on first PCA axis:")
    _ = [print(f"{feat_cols[i]}: {pca_pct.components[0, i]}") for i in idx]

    # Variance (scree) plot
    pca_pct.plot_variance()
    plt.show()

    # PCA plots: graduation ratios
    target_cols = ['cstcball_rat_grbkaat', 'cstcball_rat_grhispt',
                   'cstcball_rat_grasiat', 'cstcball_rat_gr2mort',
                   'pgcmbac_rat', 'sscmbac_rat']

    ds = Dataset.from_df(mdf, feat_cols, target_cols, test_size=0,
                         random_state=10)

    labels = ['Black', 'Hispanic', 'Asian', '2+ Races', 'Pell Grant',
              'SSL']
    ds.target_labels = [l + ' to Baseline Graduation Ratio' for l in labels]
    race_colors = get_colors(labels[:-2], tar_color_dict)
    ps_colors = get_colors(labels[-2:], tar_color_dict)
    ds.target_colors = np.append(race_colors, ps_colors, axis=0)
    pca_rat = PCAModel(ds)
    pca_rat.plot_embedding(n_obs=100, n_digits=2)
    plt.show()

    # Correlation Graphs
    columns = ['cstcball_pct_gr2mort', 'cstcball_pct_grasiat',
               'cstcball_pct_grbkaat', 'cstcball_pct_grhispt']
    labels = ['2+ Races Graduation Rate', 'Asian Graduation Rate',
              'Black Graduation Rate', 'Hispanic Graduation Rate']
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    for i, c in enumerate(columns):
        axi = ax[i//2, i % 2]
        axi.scatter(mdf['cstcball_pct_grwhitt'], mdf[c], color=colors_race[i],
                    alpha=0.3)
        axi.set_xlabel('White Graduation Rate')
        axi.set_ylabel(labels[i])
        axi.plot([0, 100], [0, 100], linestyle='--', color='black')
    plt.show()

    columns = ['pgcmbac_pct', 'sscmbac_pct']
    labels = ['Pell Graduation Rate', 'SSL Graduation Rate']
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, c in enumerate(columns):
        axi = ax[i]
        axi.scatter(mdf['nrcmbac_pct'], mdf[c], color=colors_ps[i],
                    alpha=0.3)
        axi.set_xlabel('Non-Recipient Graduation Rate')
        axi.set_ylabel(labels[i])
        axi.plot([0, 100], [0, 100], linestyle='--', color='black')
    plt.show()

