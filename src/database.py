import os
import sqlalchemy as db
import pandas as pd


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
            f'postgresql://{user}:{pwd}@{host}:{port}/gradrates'
        self.engine = db.create_engine(self.hoststring)

    def to_sql(self, df, db_table_name):
        df.to_sql(db_table_name, self.engine, if_exists='replace')

    def from_sql(self, table_name):
        # reads a SQL table into an IpedsTable object
        # see pandas.read_sql
        print(f'reading {table_name} from database')
        metadata = db.MetaData()
        table = db.Table(table_name, metadata, autoload=True, 
                         autoload_with=self.engine)
        query = db.select([table])
        return self.from_sql_query(query)

    def from_sql_query(self, sql_str):
        # reads a SQL query into a dataframe
        # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
        ResultProxy = self.engine.execute(sql_str)
        ResultSet = ResultProxy.fetchall()
        result = pd.DataFrame(data=ResultSet, columns=ResultSet[0].keys())
        # result.df.drop(columns=['index'], inplace=True)
        return result

    def close(self):
        self.engine.dispose()


if __name__ == "__main__":

    print("Connecting to database")
    ratesdb = Database(local=True)
    test_data = [[4, 5, 6], [1, 2, 3], [7, 8, 9]]
    test_df = pd.DataFrame(test_data, columns=['foo', 'bar', 'baz'])
    print("Writing to table test")
    ratesdb.to_sql(test_df, 'test')
    ratesdb.close()
