import unittest
import sys
import numpy as np
import pandas as pd
from matplotlib import cm
sys.path.append('./src')
from dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_init(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        Y = np.array([[100], [200], [300], [400]])
        Xtrain = X[[0, 3, 1], :]
        Xtest = X[2, :]
        Ytrain = Y[[0, 3, 1], :]
        Ytest = Y[2, :]

        ds = Dataset(X, Y, random_state=10)

        self.assertEqual(ds.n_features, 3)
        self.assertEqual(ds.n_targets, 1)
        self.assertEqual(ds.n_train, 3)
        self.assertEqual(ds.n_test, 1)
        self.assertTrue(np.allclose(ds.X_train, Xtrain))
        self.assertTrue(np.allclose(ds.X_test, Xtest))
        self.assertTrue(np.allclose(ds.Y_train, Ytrain))
        self.assertTrue(np.allclose(ds.Y_test, Ytest))
        self.assertTrue(np.all(np.char.equal(ds.feature_labels,
                                             np.array(['f0', 'f1', 'f2']))))
        self.assertTrue(np.all(np.char.equal(ds.target_labels,
                                             np.array(['t0']))))

    def test_from_arrays_mismatch(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        Y = np.array([[100], [200], [300]])
        self.assertRaises(ValueError, Dataset, X, Y)

    def test_from_df(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Xtrain = X[[0, 3, 1], :]
        Xtest = X[2, :]
        Ytrain = Y[[0, 3, 1], :]
        Ytest = Y[2, :]

        df1 = pd.DataFrame(X, columns=['foo', 'bar'])
        df2 = pd.DataFrame(Y, columns=['chew', 'baca'])
        df = pd.concat([df1, df2], axis=1, join='inner')
        ds = Dataset.from_df(df, ['foo', 'bar'], ['chew', 'baca'],
                             random_state=10)

        self.assertTrue(np.allclose(ds.X_train, Xtrain))
        self.assertTrue(np.allclose(ds.X_test, Xtest))
        self.assertTrue(np.allclose(ds.Y_train, Ytrain))
        self.assertTrue(np.allclose(ds.Y_test, Ytest))
        self.assertTrue(np.all(np.char.equal(ds.feature_labels,
                                             np.array(['foo', 'bar']))))
        self.assertTrue(np.all(np.char.equal(ds.target_labels,
                                             np.array(['chew', 'baca']))))

    def test_scale_features(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Xtrain = X[[0, 3, 1], :]
        Xtest = X[[2], :]
        Xtrain_sc = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0)
        Xtest_sc = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0)

        ds = Dataset(X, Y, random_state=10)
        ds.scale_features()

        self.assertIsNotNone(ds.features_scaler)
        self.assertTrue(np.allclose(ds.X_train, Xtrain_sc))
        self.assertTrue(np.allclose(ds.X_test, Xtest_sc))

    def test_unscale_features(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Xtrain = X[[0, 3, 1], :]
        Xtest = X[[2], :]

        ds = Dataset(X, Y, random_state=10)
        ds.scale_features()
        ds.unscale_features()

        self.assertIsNone(ds.features_scaler)
        self.assertTrue(np.allclose(ds.X_train, Xtrain))
        self.assertTrue(np.allclose(ds.X_test, Xtest))

    def test_scale_targets(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Ytrain = Y[[0, 3, 1], :]
        Ytest = Y[[2], :]
        Ytrain_sc = (Ytrain - Ytrain.mean(axis=0)) / Ytrain.std(axis=0)
        Ytest_sc = (Ytest - Ytrain.mean(axis=0)) / Ytrain.std(axis=0)

        ds = Dataset(X, Y, random_state=10)
        ds.scale_targets()

        self.assertIsNotNone(ds.targets_scaler)
        self.assertTrue(np.allclose(ds.Y_train, Ytrain_sc))
        self.assertTrue(np.allclose(ds.Y_test, Ytest_sc))

    def test_unscale_targets(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Ytrain = Y[[0, 3, 1], :]
        Ytest = Y[[2], :]

        ds = Dataset(X, Y, random_state=10)
        ds.scale_targets()
        ds.unscale_targets()

        self.assertIsNone(ds.targets_scaler)
        self.assertTrue(np.allclose(ds.Y_train, Ytrain))
        self.assertTrue(np.allclose(ds.Y_test, Ytest))

    def test_assign_colors(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        ds = Dataset(X, Y, random_state=10)
        ds.assign_colors()
        arr = np.array([0., 1.])
        self.assertIn('f0', ds.feature_colors.keys())
        self.assertIn('f1', ds.feature_colors.keys())
        self.assertIn('t0', ds.target_colors.keys())
        self.assertIn('t1', ds.target_colors.keys())
        self.assertTrue(np.allclose(np.array(list(ds.feature_colors.values())),
                                    cm.tab10(arr)))
        self.assertTrue(np.allclose(np.array(list(ds.target_colors.values())),
                                    cm.Dark2(arr)))

    def test_drop_features(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Xtrain = X[[0, 3, 1], :]
        Xtest = X[[2], :]
        Xtrain_sl = Xtrain[:, [1]]
        Xtest_sl = Xtest[:, [1]]

        ds = Dataset(X, Y, random_state=10)
        ds.assign_colors()
        ds.drop_features('f0')

        self.assertEquals(ds.n_features, 1)
        self.assertEquals(ds.n_targets, 2)
        self.assertIn('f1', ds.feature_labels)
        self.assertIn('f1', ds.feature_colors.keys())
        self.assertNotIn('f0', ds.feature_labels)
        self.assertNotIn('f0', ds.feature_colors.keys())
        self.assertTrue(np.allclose(ds.X_train, Xtrain_sl))
        self.assertTrue(np.allclose(ds.X_test, Xtest_sl))

    def test_drop_targets(self):
        X = np.array([[1, 3], [4, 6], [7, 9], [10, 12]])
        Y = np.array([[100, 1000], [200, 2000], [300, 3000], [400, 4000]])
        Ytrain = Y[[0, 3, 1], :]
        Ytest = Y[[2], :]
        Ytrain_sl = Ytrain[:, [1]]
        Ytest_sl = Ytest[:, [1]]

        ds = Dataset(X, Y, random_state=10)
        ds.assign_colors()
        ds.drop_targets('t0')

        self.assertEquals(ds.n_features, 2)
        self.assertEquals(ds.n_targets, 1)
        self.assertIn('t1', ds.target_labels)
        self.assertIn('t1', ds.target_colors.keys())
        self.assertNotIn('t0', ds.target_labels)
        self.assertNotIn('t0', ds.target_colors.keys())
        self.assertTrue(np.allclose(ds.Y_train, Ytrain_sl))
        self.assertTrue(np.allclose(ds.Y_test, Ytest_sl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # to run the test from the command line:
    #       python -m unittest tests/test_dataset.py
