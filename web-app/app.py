from flask import Flask, render_template, request
from formatdf import collapse_all_onehots, get_features_df
from formatdf import get_targets_df
from database import Database
# from model import Model

app = Flask(__name__)

ratesdb = Database(local=False)


@app.route('/')
def index():
    # Get names for each institution
    sql_str = 'SELECT unitid, instnm, city, stabbr ' + \
        'FROM institutions ORDER BY instnm ASC;'
    inst_df = ratesdb.from_sql_query(sql_str)
    return render_template('index.html',
                           unitid=inst_df['unitid'],
                           instnm=inst_df['instnm'],
                           city=inst_df['city'],
                           stabbr=inst_df['stabbr'],
                           len=len(inst_df['unitid']))


@app.route("/visualize", methods=['POST'])
def visualize():
    unitid = str(request.form['institution'])
    sql_str = 'SELECT * FROM institutions WHERE unitid = :unitid'
    inst_df = ratesdb.from_sql_query(sql_str, unitid=unitid)
    collapse_all_onehots(inst_df)
    features_df = get_features_df(inst_df)
    targets_df = get_targets_df(inst_df)

    return render_template('visualize.html',
                           name=inst_df.loc[0, 'instnm'],
                           city=inst_df.loc[0, 'city'],
                           stabbr=inst_df.loc[0, 'stabbr'],
                           feature_keys=list(features_df.columns),
                           feature_vals=list(features_df.values[0]),
                           feat_len=len(features_df.columns),
                           target_labels=list(targets_df.columns),
                           gr_rates=list(targets_df.values[0]),
                           tar_len=len(targets_df.columns))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=True)