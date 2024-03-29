import numpy as np
import math
import matplotlib.pyplot as plt


class Regressor(object):
    '''
    A helper class for regression-like classes
    '''

    def _fit(self, Y):

        self.model.fit(self.dataset.X_train, Y)

    def fit_train(self):

        self._fit(self.dataset.Y_train)

    def predict(self):

        self.train_predict = self.model.predict(self.dataset.X_train)
        self.train_residuals = self.dataset.Y_train - self.train_predict
        self.test_predict = self.model.predict(self.dataset.X_test)
        self.test_residuals = self.dataset.Y_test - self.test_predict

    def _rss(self):

        rss_train = np.sum(np.square(self.train_residuals), axis=0)
        rss_test = np.sum(np.square(self.test_residuals), axis=0)
        return rss_train, rss_test

    def _tss(self):

        mean_train = np.mean(self.dataset.Y_train, axis=0)
        mean_test = np.mean(self.dataset.Y_test, axis=0)
        tss_train = np.sum(np.square(self.dataset.Y_train-mean_train), axis=0)
        tss_test = np.sum(np.square(self.dataset.Y_test-mean_test), axis=0)
        return tss_train, tss_test

    def r_squared(self):

        rss_tr, rss_te = self._rss()
        tss_tr, tss_te = self._tss()
        rsq_train = 1-rss_tr/tss_tr
        rsq_test = 1-rss_te/tss_te
        return rsq_train, rsq_test

    def rmse(self, unscale=False):

        rss_tr, rss_te = self._rss()
        rmse_train = np.sqrt(rss_tr/self.dataset.Y_train.shape[0])
        rmse_test = np.sqrt(rss_te/self.dataset.Y_test.shape[0])
        if unscale:
            scale = self.dataset.targets_scaler.scale_
            rmse_train = rmse_train * scale
            rmse_test = rmse_test * scale
        return rmse_train, rmse_test

    def mae(self, unscale=False):

        mae_train = np.mean(np.abs(self.train_residuals), axis=0)
        mae_test = np.mean(np.abs(self.test_residuals), axis=0)
        if unscale:
            scale = self.dataset.targets_scaler.scale_
            mae_train = mae_train * scale
            mae_test = mae_test * scale
        return mae_train, mae_test

    def make_histograms(self, X, bins=20, x_labels=None, colors=None,
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

    def plot_residuals(self):
        '''
        graphs residuals vs predicted value for test data
        '''
        x_labels = ['Residuals: ' + l for l in self.dataset.target_labels]
        colors = [self.dataset.target_colors[l]
                  for l in self.dataset.target_labels]
        fig, ax = self.make_histograms(self.test_residuals,
                                       x_labels=x_labels, center=0,
                                       colors=colors)

    def unscale_predictions(self):

        scaler = self.dataset.targets_scaler
        self.test_predict = scaler.inverse_transform(self.test_predict)

    def unscale_residuals(self):

        scaler = self.dataset.targets_scaler
        self.test_residuals = self.test_residuals * scaler.scale_
