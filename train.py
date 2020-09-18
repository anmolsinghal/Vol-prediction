# step 1: read traininig, validation data
# Step 2: Intialise F0
# Step 3: For each Iteration i:
#   Calculate residual
#   Fit Regtree onthe residual
#   Update Fi
#   Calculate Loss
# Step 4: Pick Fk with lowest loss   
import utils
import constants
import random
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
import test

def get_residuals(truth, pred):
    residuals = {}
    for tick in truth.keys():
        residuals[tick] = truth[tick] - pred[tick] 

    return(residuals)


def get_reg_tree(residuals):
    dataset =[]
    X=[]
    for tick in residuals.keys():
        dataset.append([tick.m,tick.t, 0 if tick.cp == 'c' else 1,tick.s])
        X.append(residuals[tick])
    
    regressor = DecisionTreeRegressor(max_leaf_nodes = constants.tree_L,random_state = 0)
    dataset = np.array(dataset)
    X = np.array(X)

    regressor.fit(dataset,X)

    return(regressor)

def get_loss(ground_truth, predicted):
    loss = 0
    grid_m, grid_t = utils.create_grid()
    for tick in ground_truth.keys():
        for gm in grid_m:
            for gt in grid_t:          
                error = pow( (ground_truth[tick]- predicted[tick]),2) 
                weight = utils.loss_w(tick, gm, gt )
                loss += error*weight
    return(loss)

def get_regressor_predictions(ground_truth, regressor):
    pred = {}
    for tick in ground_truth.keys():        
        prediction = regressor.predict([[tick.m,tick.t,  0 if tick.cp == 'c' else 1, tick.s]])[0]
        pred[tick] = prediction
    return(pred)


def train():
    print("Getting ground truth")
    ground_truth = utils.get_ground_truth(constants.train_start_date, constants.train_end_date)
    validation = utils.get_ground_truth(constants.validation_start_date, constants.validation_end_date)
    print("Getting base data")
    base = get_reg_tree(ground_truth)
    print("Getting base predictions")  
    pred = get_regressor_predictions(ground_truth, base)
    print("Getting loss")
    min_loss = get_loss(ground_truth,pred)
    print("Loss at iteration 0 "+str(min_loss))
    print("Saving iteration")
    utils.save_iteration(0,base,min_loss)
    for i in range(1,constants.iterations):
        print("Getting residuals")
        residuals = get_residuals(ground_truth, pred)
        print("Getting new regressor")
        regressor = get_reg_tree(residuals)
        for tick in pred.keys():        
            prediction = regressor.predict([[tick.m,tick.t,  0 if tick.cp == 'c' else 1, tick.s]])[0]
            pred[tick] += constants.shrinkage*prediction
        print("Getting new loss")
        loss = get_loss(ground_truth,pred)
        utils.save_iteration(i,regressor,loss)
        if(loss<min_loss):
            min_loss = loss
            min_iteration = i
    
    print("Loss:"+str(min_loss))
    print("Iteration"+str(min_iteration))

def continue_train():
    ground_truth = utils.get_ground_truth(constants.train_start_date, constants.train_end_date)
    validation = utils.get_ground_truth(constants.validation_start_date, constants.validation_end_date)
    pred = {}
    for tick in ground_truth.keys():        
        prediction = test.get_prediction(tick)
        pred[tick] = prediction
    print("Getting loss")
    min_loss = get_loss(ground_truth,pred)
    print("Loss at iteration 0 "+str(min_loss))
    print("Saving iteration")
    #utils.save_iteration(0+constants.iterations,base,min_loss)
    for i in range(2*constants.iterations,3*constants.iterations):
        print("Getting residuals")
        residuals = get_residuals(ground_truth, pred)
        print("Getting new regressor")
        regressor = get_reg_tree(residuals)
        for tick in pred.keys():        
            prediction = regressor.predict([[tick.m,tick.t,  0 if tick.cp == 'c' else 1, tick.s]])[0]
            pred[tick] += constants.shrinkage*prediction
        print("Getting new loss")
        loss = get_loss(ground_truth,pred)
        utils.save_iteration(i+constants.iterations,regressor,loss)
        if(loss<min_loss):
            min_loss = loss
            min_iteration = i
    
    print("Loss:"+str(min_loss))
    print("Iteration"+str(min_iteration))

train()