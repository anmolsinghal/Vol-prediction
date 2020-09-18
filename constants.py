import datetime
from collections import namedtuple
alpha1 = 5
alpha2 = 0.5
r = 0.08
h1 = 0.5
h2 = 0.5

range = 1
shrinkage = 0.5
tree_L = 5
iterations = 300
train_start_date = datetime.datetime(2019,9,4)
train_end_date = datetime.datetime(2019,9,5)

validation_start_date = datetime.datetime(2019,9,4)
validation_end_date = datetime.datetime(2019,9,5)

trade_start_date = datetime.datetime(2019,9,4)
trade_end_date = datetime.datetime(2019,9,5)

MTA = 1

option = namedtuple("option", ["m", "t", "cp", "s"])
tick = namedtuple("tick", ["timestamp", "underlying"])

trade_data = namedtuple("trade_data", ["days","fill_px","delta","vol","gamma","theta","vega"])
tradable = namedtuple("tradable", ["cp", "strike"])
trade = namedtuple('trade', [ "delta", "pnl", "qty", "price"])

order = namedtuple('order', ["qty", "price"])