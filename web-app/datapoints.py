import pandas as pd
import json, requests
import numpy as np
import pandas as pd

class DataPoint():

    def __init__(self):
        self.URL = None
        self.Dataframe = None
        self.json_data = None
    

    def retrieve(self):
        '''
        gets the new record from the URL and preps it for the model
        '''

        self.get_json()
        self.get_df()
        self.ticket_averages()
        self.ticket_range()
        self.total_quantity_tickets()
        self.engineer_features()
        self.drop_cols()

        return self.Dataframe
       
    
    def get_json(self, URL="http://galvanize-case-study-on-fraud.herokuapp.com/data_point"):

        self.URL = URL 
        # URL  = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
        #location = "dehli technological university"
        #PARAMS = {'address':location} 

        r = requests.get(url = URL) 
        
        self.json_data = json.loads(r.text)


    def get_df(self):
        '''
        json data has to be read from dict because the nested dictionaries 
        cause a load conflict.  It has to be loaded as rows to avoid having unequal columns due to the
        nest dictionariesthen transposed to get in correct format.
        '''

        self.df = pd.DataFrame.from_dict(self.json_data,orient='index')
        self.df= self.df.T
        #copy of original made
        self.Dataframe = self.df.copy()

    def ticket_averages(self):
        '''
        ticket_types consists of 1-116 nested dictionaries.  This pulls
        the price paid for each transaction, adds it to a list, 
        and calculates the mean.
        '''
        averages = [] 
        for row in self.Dataframe['ticket_types']: 
            avg_price = []
            for i in row: 
                price = i['cost']
                avg_price.append(price)
                avg = np.mean(avg_price)
            averages.append(avg)
        rounded_averages = [np.round(row, 2) for row in averages] 
        self.Dataframe['average_ticket_price'] = rounded_averages
        self.Dataframe.fillna(0, inplace=True)

    def engineer_features(self):
        
        self.Dataframe.fillna({"delivery_method_is_0" : -9999}, inplace=True)
        self.Dataframe["event_end_min_start"] =self.Dataframe["event_end"] -self.Dataframe["event_start"]
        self.Dataframe["event_start_min_published"] =self.Dataframe["event_start"] -self.Dataframe["event_published"]
        self.Dataframe["event_end_min_published"] =self.Dataframe["event_end"] -self.Dataframe["event_published"]
        self.Dataframe["event_published_min_created"] =self.Dataframe["event_published"] -self.Dataframe["event_created"]
        self.Dataframe["event_start_min_created"] =self.Dataframe["event_start"] -self.Dataframe["event_created"]
        self.Dataframe["event_end_min_created"] =self.Dataframe["event_end"] -self.Dataframe["event_created"]
        self.Dataframe["user_turnover"] =self.Dataframe["event_created"] -self.Dataframe["user_created"]
        self.Dataframe['num_ticket_types'] = [len(i) for i in self.Dataframe['ticket_types']]
        self.Dataframe['total_prev_payouts'] = [len(i) for i in self.Dataframe['previous_payouts']]
        self.Dataframe["payout_specified"] = [0 if method == '' else 1 for method in self.Dataframe["payout_type"]]
        self.Dataframe["channel_is_0"] = [1 if channel == 0 else 0 for channel in self.Dataframe["channels"]]
        self.Dataframe["channel_is_5"] = [1 if channel == 5 else 0 for channel in self.Dataframe["channels"]]
        self.Dataframe["channel_is_6"] = [1 if channel == 6 else 0 for channel in self.Dataframe["channels"]]
        self.Dataframe["delivery_method_0"] = [1 if method == 0 else 0 for method in self.Dataframe["delivery_method"]]
        self.Dataframe.fillna(0, inplace=True)
    
    def ticket_range(self):
        ranges = [] 
        for row in self.Dataframe['ticket_types']: 
            lowest_price = []
            max_price = []
            for i in row:
                low_price = i['cost']
                if lowest_price == []:
                    lowest_price.append(low_price)
                else:
                    if low_price < lowest_price[0]:
                        lowest_price.pop(0)
                        lowest_price.append(low_price)
            for i in row:
                high_price = i['cost']
                if max_price == []:
                    max_price.append(high_price)
                else:
                    if high_price > max_price[0]:
                        max_price.pop(0)
                        max_price.append(high_price)
            range_per_row = np.array((max_price)- np.array(lowest_price))
            if range_per_row.size < 1:
                ranges.append(0)
            else:
                ranges.append(np.round(int(range_per_row)))
        
        self.Dataframe['range_ticket_price'] = ranges
    '''
        def ticket_range(self):
        ranges = [] 
        for row in self.Dataframe['ticket_types']: 
            lowest_price = []
            max_price = []
            for i in row:
                low_price = i['cost']
                if lowest_price == []:
                    lowest_price.append(low_price)
            else:
                if low_price < lowest_price[0]:
                    lowest_price.pop(0)
                lowest_price.append(low_price)
            for i in row:
                high_price = i['cost']
                if max_price == []:
                    max_price.append(high_price)
                else:
                    if high_price > max_price[0]:
                        max_price.pop(0)
                        max_price.append(high_price)
            range_per_row = np.array((max_price)- np.array(lowest_price))
            if range_per_row.size < 1:
                ranges.append(0)
            else:
                ranges.append(np.round(int(range_per_row)))
    
        self.Dataframe['range_ticket_price'] = ranges
    '''

    def total_quantity_tickets(self):
        totals = [] 
        for row in self.Dataframe['ticket_types']:
            row_totals = [] 
            for i in row: 
                dict_total = i['quantity_total'] - i['quantity_sold']
                row_totals.append(dict_total)
                sums = np.sum(row_totals)
            totals.append(sums)
        self.Dataframe['total_tickets_sold'] = totals
        #'acct_type'
    def drop_cols(self):
        self.Dataframe.drop(['approx_payout_date', 'channels', 'country',
            'currency', 'delivery_method', 'description', 'email_domain',
            'event_created', 'event_end', 'event_published', 'event_start',
            'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo',
            'listed', 'name', 'name_length', 'num_order', 'num_payouts',
            'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',
            'payee_name', 'payout_type', 'previous_payouts', 'sale_duration',
            'sale_duration2', 'ticket_types', 'user_created', 'venue_address',
            'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name',
            'venue_state', 'total_tickets_sold'], axis=1, inplace=True)


    
        