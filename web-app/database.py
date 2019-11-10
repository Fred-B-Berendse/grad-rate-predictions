import os
import sqlalchemy as db
from sqlalchemy import text
import pandas as pd
from formatdf import collapse_all_onehots, get_features_df
from formatdf import get_targets_df


class Database(object):

    def __init__(self, local=True):

        if local:
            host = 'localhost'
            port = '5435'
            user = 'postgres'
            pwd = ''
        else:
            host = 'database-1.cwtci684dbqt.us-west-1.rds.amazonaws.com'
            port = '5432'
            user = os.environ.get('AWS_RDS_USERNAME')
            pwd = os.environ.get('AWS_RDS_PASSWORD')
        self.hoststring = \
            'postgresql://{}:{}@{}:{}/gradrates'.format(user, pwd, host, port)
        self.engine = db.create_engine(self.hoststring)

    def to_sql(self, df, db_table_name):
        df.to_sql(db_table_name, self.engine, if_exists='replace')

    def from_sql(self, table_name):
        # reads a SQL table into a Pandas dataframe 
        # see pandas.read_sql
        print('reading {} from database'.format(table_name))
        metadata = db.MetaData()
        table = db.Table(table_name, metadata, autoload=True, 
                         autoload_with=self.engine)
        query = db.select([table])
        return self.from_sql_query(query)

    def from_sql_query(self, sql_str, **kwargs):
        # reads a SQL query into a dataframe
        # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
        sql = text(sql_str)
        ResultProxy = self.engine.execute(sql, **kwargs)
        ResultSet = ResultProxy.fetchall()
        result = pd.DataFrame(data=ResultSet, columns=ResultSet[0].keys())
        return result

    def close(self):
        self.engine.dispose()


if __name__ == "__main__":

    print("Connecting to database")
    ratesdb = Database(local=False)
    unitid = 188429
    sql_str = 'SELECT * FROM institutions WHERE unitid = :unitid'
    inst_df = ratesdb.from_sql_query(sql_str, unitid=unitid)

    collapse_all_onehots(inst_df)
    features_df = get_features_df(inst_df)
    targets_df = get_targets_df(inst_df)

