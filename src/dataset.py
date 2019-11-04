import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import cm


class DropError(Exception):
    pass


class Dataset(object):

    def __init__(self, X, Y, test_size=0.25, random_state=None,
                 feature_labels=None, target_labels=None):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Dimensions of X and Y do not match')

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X, Y, test_size=test_size,
                             random_state=random_state)
        self.n_features = self.X_train.shape[1]
        self.n_targets = self.Y_train.shape[1]
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]

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

        self.feature_colors = None
        self.target_colors = None
        self.features_scaler = None
        self.targets_scaler = None

    @classmethod
    def from_df(cls, df, feature_cols, target_cols, test_size=0.25,
                random_state=None):
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
        self.n_features -= 1
        mask = ~np.isin(self.feature_labels, feature_labels)
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
        self.n_targets -= 1
        mask = ~np.isin(self.target_labels, target_labels)
        sl = self.Y_train[:, mask]
        self.Y_train = sl.reshape(self.n_train, self.n_targets)
        if self.Y_test is not None:
            sl = self.Y_test[:, mask]
            self.Y_test = sl.reshape(self.n_test, self.n_targets)
        self.target_labels = self.target_labels[mask]
        if self.target_colors is not None:
            [self.target_colors.pop(l) for l in target_labels]

    def transform_feature(self, label, new_label, function, drop_old=True):
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
        features_dict: {feature_label: (new_label_name, function)}
        '''
        for label, (new_label, func) in features_dict.items():
            self.transform_feature(label, new_label, func, drop_old=drop_old)

    def transform_target(self, label, new_label, function, drop_old=True):
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
        targets_dict: {target_label: (new_label_name, function)}
        '''
        for label, (new_label, func) in targets_dict.items():
            self.transform_target(label, new_label, func, drop_old=drop_old)
