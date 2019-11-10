from flask import Flask, render_template, request
import numpy as np
import pandas as pd
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
    return render_template('visualize.html', data=inst_df)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=True)