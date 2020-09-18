import utils
import constants
import pickle
import datetime
import os
import sys

from py_vollib.black_scholes import black_scholes

#helper function to get prediction of implied volatility
def get_prediction(tick):
    pred = 0
    for i in range(0,399):
        if i%100==0:
            continue
        file = 'model_'+str(i)+'.sav' 
        model = pickle.load(open('regressors/'+file,'rb'))
        if file != 'model_0.sav' :
            pred += constants.shrinkage*model.predict([[tick.m,tick.t,  0 if tick.cp == 'c' else 1, tick.s]])[0]
        else:
            pred += model.predict([[tick.m,tick.t,  0 if tick.cp == 'c' else 1, tick.s]])[0]
    return(pred)


gt= utils.get_ground_truth(constants.train_start_date, constants.train_start_date)
for tick in gt.keys():
    print("Ground truth vol: "+str(gt[tick])+" predicted vol: "+str(get_prediction(tick)))
    #sleep(2)


#helper function to read market data to trade
def get_trade_data():
    cur_date= constants.trade_start_date
    ground_truth = {}
    
    while(cur_date<= constants.trade_end_date):
        data = utils.read_day_file(cur_date)
        day = {}
        for index, row in data.iterrows():
            timestamp = row["time"]
            underlying= float(row["underlying_price"])
            
            ticker = constants.tick(timestamp= timestamp, underlying = underlying)

            
            op = constants.tradable(strike = float(row["strike"]),  cp = 'c' if row["type"] =="call" else 'p')
                        
            
            val = constants.trade_data( days = float(row["days_to_expiry"]),fill_px = (row["ask_px"]+ row["bid_px"])/2, delta = row["delta"], vol = row["volatility"], gamma = row["gamma"], theta= row["theta"], vega = row["vega"])
            if ticker in day.keys():
                day[ticker].update({op:val})
            else:
                day[ticker] = {}
                day[ticker].update({op:val})
        ground_truth[cur_date] = day
        cur_date = utils.get_next_day(cur_date)
    return(ground_truth)

#helper function to get profitable opportunities linearly scaled in PnL based on price arbitrarge with volatility prediction 
def get_dNeutral_positions(ticker, underlying):
    positions = {}
    min_pnl = sys.maxsize
    for tick in ticker.keys():
        m = tick.strike/underlying
        t = ticker[tick].days/365
        cp = tick.cp
        s = underlying

        td = tick
        fair_price = black_scholes(cp,s,m*s,t,0.08,get_prediction(constants.option(m=m,s=s, t=t,cp=cp))) 
        fill_price = ticker[tick].fill_px
        pnl = abs(fill_price-fair_price)
        if fill_price == fair_price:
            continue
        
        if fill_price > fair_price:
            opp = constants.trade( delta = ticker[tick].delta, pnl = pnl, qty=-1, price= fill_price)
        else:
            opp = constants.trade( delta = ticker[tick].delta, pnl = pnl,qty= 1, price= -1*fill_price)
        if pnl < min_pnl:
            min_pnl = pnl
        
        positions.update({td:opp})
   
    for pos in positions.keys():
        positions[pos]=constants.trade(delta = positions[pos].delta, pnl = positions[pos].pnl,qty= int(positions[pos].qty*positions[pos].pnl/min_pnl), price = positions[pos].price)
    return(positions)

#helper function to transition positions in options        
def get_diff_trades(new_portfolio,old_portfolio ):
    trades = {}
    
    for td in new_portfolio.keys():
        if td in old_portfolio.keys():
            qty_change = new_portfolio[td].qty - old_portfolio[td].qty
            if qty_change != 0:
                trades.update({td:constants.order(qty = qty_change, price = new_portfolio[td].price)})
        else:
            trades.update({td:constants.order(qty = new_portfolio[td].qty, price = new_portfolio[td].price)})
    return(trades)

#get arbitrarge based trades due to volatility prediction and hedge delta using underlying
def get_delta_neutral_trades(ticker, underlying_qty, underlying_price, cur_portfolio ):
    positions = get_dNeutral_positions(ticker, underlying_price)
    underlying = 0
    
    for pos in positions.keys():
        underlying += positions[pos].delta*positions[pos].qty
    
    trades = get_diff_trades(positions, cur_portfolio)
    underlying_change= underlying - underlying_qty
    cash_change = -1*underlying_change*underlying_price
    for pos in trades.keys():
        cash_change += -1*trades[pos].price*trades[pos].qty
        
    return trades, underlying_change, cash_change

#helper function to update portfolio based on trades recommended
def update_portfolio(cur_portfolio, trades, d_trades):
    for td in trades.keys():
        if td in cur_portfolio.keys():
            cur_portfolio[td] = constants.order(qty= cur_portfolio[td].qty+ trades[td].qty, price = trades[td].price)
        else:
            cur_portfolio.update({td:constants.order(qty= trades[td].qty, price = trades[td].price)})
    
    for td in d_trades.keys():
        if td in cur_portfolio.keys():
            cur_portfolio[td] = constants.order(qty= cur_portfolio[td].qty+ d_trades[td].qty, price = d_trades[td].price)
        else:
            cur_portfolio.update({td:constants.order(qty= trades[td].qty, price = trades[td].price)})        
    return cur_portfolio

#get trades based on predicted option price movement due to change in underlying and in porcess add delta to portfolio
def get_delta_trades(ticker, prev_underlying, underlying):
    d_underlying = underlying - prev_underlying
    min_pnl = sys.maxsize
    trades = {}
    cash = 0
    for tick in ticker:
        fill_price = ticker[tick].fill_px
        delta_px = ticker[tick].delta
        td_px = fill_price + delta_px

        pnl = td_px - fill_price

        trades.update({tick:constants.order(qty = pnl, price = fill_price)})
        if pnl < min_pnl:
            min_pnl = pnl
    for tick in trades:
        trades[tick] =constants.order(qty = int(trades[tick].qty/min_pnl), price = trades[tick].price) 
        cash += -1*trades[tick].price*trades[tick].qty
    return(trades, cash)

#helper function to print portfolio value with given ticker data
def get_portfolio_value(ground_truth,portfolio, cash, underlying_price, underlying_qty ):
    value = cash + underlying_price*underlying_qty
    for td in portfolio:
        if td in ground_truth.keys():
            value += -1*ground_truth[td].fill_px*portfolio[td].qty
        else:
            value += -1*portfolio[td].price*portfolio[td].qty
    return(value)
#main function to read data and start trading, updating portfolio and printing value
def trade():
    ground_data = get_trade_data()
    cur_portfolio = {}
    cash = 0
    underlying = 0
    prev_underlying = None
    for day in ground_data.keys():
        for time in ground_data[day].keys():
            old_value = get_portfolio_value(ground_data[day][time],cur_portfolio,cash, time.underlying,underlying)
            dNeutral_trades, underlying_change, cash_change = get_delta_neutral_trades(ground_data[day][time], underlying,time.underlying, cur_portfolio)
            if prev_underlying != None:
                d_trades, cash_dchange = get_delta_trades(ground_data[day][time], prev_underlying, time.underlying)
            else:
                d_trades= {}
                cash_dchange= 0
            cur_portfolio = update_portfolio( cur_portfolio,dNeutral_trades, d_trades)
            underlying += underlying_change
            cash += cash_change + cash_dchange
            prev_underlying = time.underlying
            new_value = get_portfolio_value(ground_data[day][time],cur_portfolio,cash, time.underlying,underlying)
            print("Old "+"Rs{:,.2f}".format(old_value)+" New "+"Rs{:,.2f}".format(new_value))
          
    
