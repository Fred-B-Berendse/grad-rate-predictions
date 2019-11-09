import pickle
# from xgboost import XGBClassifier, plot_tree, plot_importance

class Model(object):

    def __init__(self):
        self.model = None

    def load_model(self, pkl_filepath):
        # loaded_model = pickle.load(open('..src/models/model_all_features_fin.pkl', 'rb'))
        self.model = pickle.load(open(pkl_filepath, 'rb'))

    def preds_new_data_point(self, new_data):
        y_pred = self.model.predict(new_data)
        y_prob = self.model.predict_proba(new_data)[0,1]
        return int(y_pred), float(y_prob)