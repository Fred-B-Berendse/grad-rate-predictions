import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import cm


class DropError(Exception):
    pass


class Dataset(object):
    '''
    A class for handling features and targets used in model.
    '''

    def __init__(self, X, Y, test_size=0.25, random_state=None,
                 feature_labels=None, target_labels=None):

        # Raise an error if the lengths of X and Y do not match
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Dimensions of X and Y do not match')

        # Perform a test-train split
        idx = np.array(range(X.shape[0]))
        self.X_train, self.X_test, \
            self.Y_train, self.Y_test, \
            self.idx_train, self.idx_test = \
            train_test_split(X, Y, idx, test_size=test_size,
                             random_state=random_state)

        # Save counts as attributes
        self.n_features = self.X_train.shape[1]
        self.n_targets = self.Y_train.shape[1]
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]

        # Save feature and target labels if given
        # otherwise refer to defaults f0, f1, f2... and t0, t1, t2
        if feature_labels is None:
            self.feature_labels = np.array(
                                    ['f'+str(i) for i in range(X.shape[1])])
        else:
            self.feature_labels = feature_labels
        if target_labels is None:
            self.target_labels = np.array(
                                    ['t'+str(j) for j in range(Y.shape[1])])
        else:
            self.target_labels = target_labels

        # Initialize feature/target colors and scalers
        self.feature_colors = None
        self.target_colors = None
        self.features_scaler = None
        self.targets_scaler = None

    @classmethod
    def from_df(cls, df, feature_cols, target_cols, test_size=0.25,
                random_state=None):
        '''
        Instantiates the class from a pandas Dataframe.
        - feature_cols is the list of column names to become features
        - target_cols is the list of columns names to become targets
        '''
        X = df.loc[:, feature_cols].values
        Y = df.loc[:, target_cols].values
        return cls(X, Y, test_size=test_size, random_state=random_state,
                   feature_labels=feature_cols, target_labels=target_cols)

    @staticmethod
    def _scale(train, test):
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        if test is not None and len(test) > 0:
            test = scaler.transform(test)
        return scaler, train, test

    def scale_features(self):
        sc, tr, te = self._scale(self.X_train, self.X_test)
        self.features_scaler = sc
        self.X_train = tr
        self.X_test = te

    def scale_targets(self):
        sc, tr, te = self._scale(self.Y_train, self.Y_test)
        self.targets_scaler = sc
        self.Y_train = tr
        self.Y_test = te

    def scale_features_targets(self):
        self.scale_features()
        self.scale_targets()

    @staticmethod
    def _unscale(scaler, train, test):
        tr = scaler.inverse_transform(train)
        te = scaler.inverse_transform(test)
        return tr, te

    def unscale_features(self):
        tr, te = self._unscale(self.features_scaler,
                               self.X_train,
                               self.X_test)
        self.X_train = tr
        self.X_test = te
        self.features_scaler = None

    def unscale_targets(self):
        tr, te = self._unscale(self.targets_scaler,
                               self.Y_train,
                               self.Y_test)
        self.Y_train = tr
        self.Y_test = te
        self.targets_scaler = None

    def unscale_features_targets(self):
        self.unscale_features()
        self.unscale_targets()

    def assign_colors(self, feature_cmap=cm.tab10, target_cmap=cm.Dark2):
        fc = feature_cmap(np.linspace(0, 1, len(self.feature_labels)))
        self.feature_colors = dict(zip(self.feature_labels, fc))
        tc = target_cmap(np.linspace(0, 1, len(self.target_labels)))
        self.target_colors = dict(zip(self.target_labels, tc))

    def drop_features(self, feature_labels):
        if self.features_scaler is not None:
            raise DropError('Cannot drop features when scaled.')
        if type(feature_labels) == str:
            feature_labels = [feature_labels]
        mask = ~np.isin(self.feature_labels, feature_labels)
        self.n_features = mask.sum()
        sl = self.X_train[:, mask]
        self.X_train = sl.reshape(self.n_train, self.n_features)
        if self.X_test is not None:
            sl = self.X_test[:, mask]
            self.X_test = sl.reshape(self.n_test, self.n_features)
        self.feature_labels = self.feature_labels[mask]
        if self.feature_colors is not None:
            [self.feature_colors.pop(l) for l in feature_labels]

    def drop_targets(self, target_labels):
        if self.targets_scaler is not None:
            raise DropError('Cannot drop targets when scaled.')
        if type(target_labels) == str:
            target_labels = [target_labels]
        mask = ~np.isin(self.target_labels, target_labels)
        self.n_targets = mask.sum()
        sl = self.Y_train[:, mask]
        self.Y_train = sl.reshape(self.n_train, self.n_targets)
        if self.Y_test is not None:
            sl = self.Y_test[:, mask]
            self.Y_test = sl.reshape(self.n_test, self.n_targets)
        self.target_labels = self.target_labels[mask]
        if self.target_colors is not None:
            [self.target_colors.pop(l) for l in target_labels]

    def transform_feature(self, label, new_label, function, drop_old=True):
        '''
        Applies a function to every value in the given feature. The result is
        saved as a new feature.

            label - name of the input feature
            new_label - name of the new output feature
            function - a function to apply to each value in the feature
        '''
        idx = np.where(self.feature_labels == label)
        self.feature_labels = np.append(self.feature_labels, new_label)
        self.n_features += 1
        tr = function(self.X_train[:, idx]).flatten().reshape(-1, 1)
        self.X_train = np.append(self.X_train, tr, axis=1)
        te = function(self.X_test[:, idx]).flatten().reshape(-1, 1)
        self.X_test = np.append(self.X_test, te, axis=1)
        if self.feature_colors is not None:
            self.feature_colors[new_label] = self.feature_colors[label]
        if drop_old:
            self.drop_features(label)

    def transform_features(self, features_dict, drop_old=True):
        '''
        Transforms multiple features by applying a function to each value.
        The results are saved as new features.

            features_dict - a dictionary with the following format:
                            {feature_label: (new_label_name, function)}
        '''
        for label, (new_label, func) in features_dict.items():
            self.transform_feature(label, new_label, func, drop_old=drop_old)

    def transform_target(self, label, new_label, function, drop_old=True):
        '''
        Applies a function to every value in the given target. The result is
        saved as a new target.

            label - name of the input feature
            new_label - name of the new output feature
            function - a function to apply to each value in the feature
        '''
        idx = np.where(self.target_labels == label)
        self.target_labels = np.append(self.target_labels, new_label)
        self.n_targets += 1
        tr = function(self.Y_train[:, idx]).flatten().reshape(-1, 1)
        self.Y_train = np.append(self.Y_train, tr, axis=1)
        te = function(self.Y_test[:, idx]).flatten().reshape(-1, 1)
        self.Y_test = np.append(self.Y_test, te, axis=1)
        if self.target_colors is not None:
            self.target_colors[new_label] = self.target_colors[label]
        if drop_old:
            self.drop_targets(label)

    def transform_targets(self, targets_dict, drop_old=True):
        '''
        Transforms multiple features by applying a function to each value.
        The results are saved as new features.

            features_dict - a dictionary with the following format:
                            {target_label: (new_label_name, function)}
        '''
        for label, (new_label, func) in targets_dict.items():
            self.transform_target(label, new_label, func, drop_old=drop_old)

    def make_feature_histograms(self, features, x_labels=None, colors=None,
                                center=None):
        '''
        Makes histograms of all features listed

            features - list of features to graph
            x_labels - list of labels to use for each x-axis
            colors - list of colors to use for each feature
        '''
        indexes = [np.where(self.feature_labels == f) for f in features]
        indexes = np.array(indexes).flatten()
        self._make_histograms(self.X_train[:, indexes], x_labels=x_labels,
                              colors=colors, center=center)

    def _make_histograms(self, X, bins=20, x_labels=None, colors=None,
                         center=None):
        nplots = X.shape[1]
        nplotrows = math.ceil(nplots/2)
        if colors is None:
            colors = np.repeat(None, nplots)
        fig, ax = plt.subplots(nplotrows, 2, figsize=(12, 4*nplotrows))
        for i in range(nplots):
            axi = ax[i] if nplots <= 2 else ax[i//2, i % 2]
            data = X[:, i]
            axi.hist(data, bins=bins, color=colors[i], alpha=0.8)
            if x_labels is not None:
                axi.set_xlabel(x_labels[i])
            if center is not None:
                rng = max(center-min(data), max(data)-center)*1.05
                axi.set_xlim(center-rng, center+rng)
        plt.tight_layout()
        return fig, ax

    def log10_sm(self, x):
        return np.log10(x + 1)

    def log10u_sm(self, x):
        return np.log10(101-x)

    @staticmethod
    def validname(name):
        res = name.replace('-', '_').replace('+', 'plus')
        res = res.replace(' ', '_').replace('2', 'two')
        return res.lower()
