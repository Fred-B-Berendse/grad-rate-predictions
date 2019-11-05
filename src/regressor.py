import numpy as np
from plotting import make_histograms


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

    def plot_residuals(self):
        '''
        graphs residuals vs predicted value for test data
        '''
        x_labels = ['Residuals: ' + l for l in self.dataset.target_labels]
        colors = [self.dataset.target_colors[l]
                  for l in self.dataset.target_labels]
        fig, ax = make_histograms(self.test_residuals,
                                  x_labels=x_labels, center=0,
                                  colors=colors)
