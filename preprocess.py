import utils
import datetime
import numpy as np
import pandas as pd
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes import implied_volatility


cur_date = datetime.datetime(2019,9,4)
end_date = datetime.datetime(2019,10,4)

#helper function to pivot data
def pivot_process(start_date, end_date):
    while(cur_date<= end_date):
        data = utils.read_day_file(cur_date)
        BS_price = []
        data['cp'] = data.type.str.slice(start =0, stop = 1)
        for index, row in data.iterrows():
            BS_price.append(black_scholes(row.cp,row.underlying_price,row.strike, row.days_to_expiry/365, 0.08, row.volatility))
        data["BS_price"] = BS_price
        table = pd.pivot_table(data, values = ['underlying_price', 'bid_px','ask_px', 'trade_px', 'BS_price'], index = ['strike', 'days_to_expiry', 'type'], aggfunc = np.average)
        table.to_csv('trial.csv', index = True)
        table = pd.read_csv('trial.csv')
    
        table['cp'] = table.type.str.slice(start =0, stop = 1)
        vol = []
        err = 0
        for index, row in table.iterrows():
            try:
                vol.append(implied_volatility.implied_volatility(row.BS_price, row.underlying_price, row.strike, row.days_to_expiry/365, 0.08, row.cp))
            except:
                print(row)
                err+=1
                vol.append(0)
                continue
        table['volatility'] = vol
        table.to_csv("processed/data_"+cur_date.strftime("%Y")+cur_date.strftime("%m")+cur_date.strftime("%d")+".csv", index = True)

        cur_date = utils.get_next_day(cur_date)

#helper function to sample data
def sample_process(cur_date, end_date):
    while(cur_date<= end_date):
        data = utils.read_day_file(cur_date)
        strikes = set(data['strike'])
        strikes= list(strikes)
        appended_data = []
        for strike in strikes:
            filtered = data.loc[data['strike']==strike]
            sampled = filtered.sample(frac=0.1, random_state=1)
            appended_data.append(sampled)
        appended_data = pd.concat(appended_data)
        appended_data.to_csv("sampled/data_"+cur_date.strftime("%Y")+cur_date.strftime("%m")+cur_date.strftime("%d")+".csv")
        cur_date = utils.get_next_day(cur_date)
