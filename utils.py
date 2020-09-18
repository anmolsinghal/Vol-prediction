import csv
import datetime
import pandas as pd
import math
import constants
import os
import numpy as np
import json
import pickle

#helper function to get next day 
def get_next_day(date):
    files = os.listdir("banknifty/")
    date = date + datetime.timedelta(days=1)
    while "data_"+date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")+".csv" not in files and date< datetime.datetime(2019,10,4):
        date = date + datetime.timedelta(days=1)
    return(date)

#helper function to read data from csv files
def read_day_file(date):
    data = pd.read_csv("banknifty/data_"+date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")+".csv")
    data['time']= data['time'].str.slice(0,5)
    pd.to_datetime(data['time'], format='%H:%M')
    return(data)

#helper function to read training data
def get_tick_as_dict(row):
    m = float(row["strike"])/float(row["underlying_price"])
    t = float(row["days_to_expiry"])/365
    cp = 'c' if row["type"] =="call" else 'p'
    s = float(row["underlying_price"])

    
    op = constants.option(m = m, s=s, cp=cp,t=t)
    return({op: float(row["volatility"])})

#helper function to read training data
def get_ground_truth(start_date, end_date):
    cur_date= start_date
    ground_truth = {}
    while(cur_date<= end_date):
        data = read_day_file(cur_date)
        for index, row in data.iterrows():
            op = get_tick_as_dict(row)
            ground_truth.update(op)

        cur_date = get_next_day(cur_date)
    return(ground_truth)
#helper function for loss calculation
def create_grid():
    output_m= np.linspace(0.2,2,num = 15)
    output_t = np.linspace(1/365,3,num = 15)    
    return((output_m, output_t))
#helper function for loss calculation
def loss_k(u, v):
    return( (0.5/math.pi)*math.exp(-0.5*( u*u + v*v)) )

def loss_w1(op):
    if op.cp == 'c':
        return(1/math.pi*math.atan(constants.alpha1*(op.m-1))+0.5)
    else:
        return(1/math.pi*math.atan(constants.alpha1*(1-op.m))+0.5)
#helper function for loss calculation
def loss_w2(op):
    return(1/math.pi*math.atan(constants.alpha2*(1-op.t))+0.5)

#helper function for loss calculation
def loss_w( pred, gm, gt ):
    return (loss_w1(pred)*loss_w2(pred)*loss_k( ((gm-pred.m)/constants.h1),((gt-pred.t)/constants.h2)))

#save iteration data while training
def save_iteration(i,regressor, loss):
    filename = 'regressors2/model_'+str(i)+'.sav'
    pickle.dump(regressor, open(filename, 'wb'))

    loss_file = open("loss_file2.txt", "a")
    loss_file.writelines(["Loss for iteration"+str(i)+": "+str(loss)+'\n'])
    print("Loss for iteration"+str(i)+": "+str(loss)+'\n')
    loss_file.close()

