import os
import sqlalchemy as db
import pandas as pd


class Database(object):
    '''
    A class for handling database connections and queries
    '''

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
            f'postgresql://{user}:{pwd}@{host}:{port}/gradrates'
        self.engine = db.create_engine(self.hoststring)

    def to_sql(self, df, db_table_name):
        '''
        writes the contents of a pandas dataframe to the database table
        db_table_name
        '''
        df.to_sql(db_table_name, self.engine, if_exists='replace')

    def from_sql(self, table_name):
        '''
        Reads a database table into a pandas DataFrame
        '''
        print(f'reading {table_name} from database')
        metadata = db.MetaData()
        table = db.Table(table_name, metadata, autoload=True, 
                         autoload_with=self.engine)
        query = db.select([table])
        return self.from_sql_query(query)

    def from_sql_query(self, sql_str):
        '''
        Executes a SQL string and returns the results as a pandas DataFrame
        '''
        # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
        ResultProxy = self.engine.execute(sql_str)
        ResultSet = ResultProxy.fetchall()
        result = pd.DataFrame(data=ResultSet, columns=ResultSet[0].keys())
        return result

    def close(self):
        self.engine.dispose()
