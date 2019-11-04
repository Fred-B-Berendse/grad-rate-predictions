import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from numpy.random import choice
from sklearn.decomposition import PCA
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


def make_histograms(X, bins=20, x_labels=None, colors=None, center=None):

    nplots = X.shape[1]
    nplotrows = math.ceil(nplots/2)
    if colors is None:
        # colors = cm.tab10(np.linspace(0, 1, nplots))
        colors = np.repeat(None, nplots)
    fig, ax = plt.subplots(nplotrows, 2, figsize=(12, 4*nplotrows))
    for i in range(nplots):
        axi = ax[i] if nplots <= 2 else ax[i//2, i % 2]
        data = X[:, i]
        axi.hist(data, bins=bins, color=colors[i])
        if x_labels is not None:
            axi.set_xlabel(x_labels[i])
        if center is not None:
            rng = max(center-min(data), max(data)-center)*1.05
            axi.set_xlim(center-rng, center+rng)
    plt.tight_layout()
    return fig, ax


def make_scatterplots(X, Y, x_labels=None, y_labels=None, colors=None,
                      ycenter=None):

    nplots = X.shape[1]
    nplotrows = math.ceil(nplots/2)
    if colors is None:
        # colors = cm.tab10(np.linspace(0, 1, nplots))
        colors = np.repeat(None, nplots)
    fig, ax = plt.subplots(nplotrows, 2, figsize=(12, 4*nplotrows))
    for i in range(nplots):
        axi = ax[i//2, i % 2]
        ydata = Y[:, i]
        axi.scatter(X[:, i], ydata, color=colors[i])
        if x_labels is not None:
            axi.set_xlabel(x_labels[i])
        if y_labels is not None:
            axi.set_ylabel(y_labels[i])
        if ycenter is not None:
            rng = max(ycenter-min(ydata), max(ydata)-ycenter)*1.05
            axi.set_ylim(ycenter-rng, ycenter+rng)
    plt.tight_layout()
    return fig, ax


def make_heatmap(arr, x_labels=None, y_labels=None, cmap='tab10',
                 center=None):

    ax = sns.heatmap(arr, xticklabels=x_labels, yticklabels=y_labels,
                     cmap=cmap, annot=True, center=0, annot_kws={"size": 16},
                     fmt='.2f')
    return ax


def make_barplot(X, x_labels=None, y_label=None, color=None, width=0.8):

    nbars = len(X)
    x_loc = np.linspace(0, 1.2*width*nbars, nbars)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.bar(x_loc, X, width=width, color=color, alpha=0.8)
    ax.set_xlim(x_loc[0]-0.6*width, x_loc[-1]+0.6*width)
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    return fig, ax


def make_stacked_barplot(X_arr, x_labels=None, y_label=None, colors=None,
                         stack_labels=None, width=0.8):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    n_bars = len(X_arr)
    x = np.linspace(0, 1.2*width*n_bars, n_bars)
    for i in range(n_bars):
        lab = None if i > 0 else stack_labels
        make_stacked_bar(x[i], X_arr[i], colors, ax, labels=lab, width=width)
    ax.set_xlim(x[0]-0.6*width, x[-1]+0.6*width)
    ax.set_xticks(x)
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.legend(loc='best', frameon=True)
    return fig, ax


def make_stacked_bar(x, y_arr, colors, ax, labels=None, width=1):

    if labels is None:
        labels = [None for _ in range(len(y_arr))]
    prev_cnt = 0
    for i, y in enumerate(y_arr):
        la = labels[i] if labels is not None else None
        c = colors[i] if colors is not None else None
        ax.bar(x, y, width=width, bottom=prev_cnt, label=la,
               color=c, alpha=0.6)
        prev_cnt += y
    return


def make_violin_plots(df, columns, col_labels, colors=None, colormap=cm.tab10):

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if colors is None:
        colors = colormap(np.linspace(0, 1, len(columns)))
    dataset = []
    for c in columns:
        dataset.append(df.loc[:, c].dropna().values)

    parts = ax.violinplot(positions=range(len(col_labels)),
                          dataset=dataset,
                          showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    return fig, ax, colors


def make_rank_plot(df, x_cols, y_avg, titles):

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i, (x, title) in enumerate(zip(x_cols, titles)):
        df.plot(x, y_avg, kind='scatter', ax=ax[i], label='Avg Both Tests')
        df.plot(x, x+'_scl', kind='scatter', ax=ax[i], label='One Test',
                color='red')
        ax[i].legend(loc='best')
        ax[i].set_xlabel('Benchmark Score')
        ax[i].set_ylabel('Institution Percentile')
        ax[i].set_title(title)
    return fig, ax


def scree_plot(pca, n_components_to_plot=8, title=None):

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.1f}%".format(vals[i]),
                    (ind[i]+0.2, vals[i]+0.5),
                    va="bottom",
                    ha="center",
                    fontsize=12)

    ax.set_xticks(ind)
    ax.set_xticklabels(ind+1, fontsize=12)
    ax.set_ylim(0, max(vals) + 5)
    ax.set_xlim(0 - 0.5, n_components_to_plot - 0.5)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    return fig, ax


def make_embedding_graph(X, y, topic, n=-1):

    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if n > 0:
        sample = choice(range(len(y)), n, replace=False)
        X_plot, y_plot = X_pca[sample], y[sample]
    else:
        X_plot, y_plot = X, y
    plot_embedding(ax, X_plot, y_plot, title='Principal Component Embedding: '
                   + topic)
    return fig, ax


def scale(y, n_sigma):
    '''
    Scales y values to range between 0 and 1. 
    Values at the median scale to 0.5 
    Clips values n_sigma standard deviations above and below the median. 
    '''
    yscl = (y - np.median(y))/(n_sigma*y.std()) + 0.5
    return np.clip(yscl, 0, 1)


def plot_embedding(ax, X, y, title=None):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.patch.set_visible(False)
    ysc = scale(y, 1.5)
    for i in range(X.shape[0]):
        
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=cm.coolwarm(ysc[i]),
                 fontdict={'weight': 'bold', 'size': 12})

    ax.set_xticks([]),
    ax.set_yticks([])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.1, 1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)


def calc_partial_dependence(estimator, X, feature, n=10):
    '''
    Caculates the partial dependence of X[feature] on the estimator prediction
    '''
    Xc = X.copy()
    x_feat = Xc[:, feature]
    x_feat.sort()
    x_feat = np.unique(x_feat)
    result = []
    feature_values = sample_array(x_feat, n)
    for v in feature_values:
        Xc[:, feature] = v
        y_pred = estimator.predict(Xc)
        result.append(y_pred.mean())
    return feature_values, np.array(result)


def plot_partial_dependence(estimator, X, feature, ax, n_points=10,
                            color=None, label=None):
    '''
    Plots partial dependence of X[feature] using the provided estimator
    '''
    feat_vals, part_dep = calc_partial_dependence(estimator,
                                                  X,
                                                  feature,
                                                  n=n_points)
    ax.plot(feat_vals, part_dep, color=color, label=label)
    ax.set_ylabel('Mean Prediction')


def sample_array(arr, n):
    '''
    Gets n samples evenly spaced samples (by index number) from the array arr
    '''
    step = len(arr) // (n - 1)
    if step == 0:
        return np.array(arr)
    idx = np.arange(0, len(arr), step=step, dtype=int)
    return np.array([arr[i] for i in idx])


def make_color_dict(labels, cmap, start=0, end=1):
    '''
    Makes a dictionary of colors for each key in keys
    '''
    colors = cmap(np.linspace(start, end, len(labels)))
    return dict(zip(labels, colors))


def get_colors(labels, color_dict):
    '''
    Extracts a list of colors from a color dictionary in the order of labels
    '''
    results = []
    [results.append(color_dict[l]) for l in labels]
    return np.array(results)
