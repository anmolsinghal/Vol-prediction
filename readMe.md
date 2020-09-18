# Description
Python implementation of [Semi parametric forecasts of the implied volatility surface using regression trees](https://www.researchgate.net/publication/220286541_Semi-parametric_forecasts_of_the_implied_volatility_surface_using_regression_trees)

# Dependencies
Scikit-learn <br/>
python-vollib <br/>
numpy <br/>
pickle <br/>
pandas <br/>

# Implementation
We train a TreeFGD model on predicting implied volatility of an option. This implied volatility prediction process is described in the paper.

# Data
The data is banknifty data over period of one month with one minute intervals. The data can be preprocessed to reduce sizxe for training by calling the function in preprocess.py. 
Dont change the directory structure as provided as paths are hardcoded for now.

# Training the model
Train.py has all relevant functions to train the model and save it for testing. To run training in your python script  
~~~
import train
train.train()
~~~
You can modify training parameters from constants.py 
Utils.py has a bunch of utility function used while training/testing the model 
train.py has functions used to train the model with a explanatory commented header 

## Testing the model
Test.py uses the saved model to generate predictions for true implied volatility of the option and hence the fair price using Black Scholes model. 
This fair price is used to identify delta neutral opportunities of arbitrarge used to maximise profit. Opportunities are linearly scaled in their PnL impact to determine position size and create a delta neutral portfolio. 
To hedge the delta, we trade the underlying.  

Since option prices follow stock prices, and the relation of the option price to its underlying price is given by delta, we can safely predict the option price movement in next ticker based on underlying price movement in the current one. 
This information has been used to introduce delta trades in the activity to provide a safe and guaranteed profit.  

To test the model, run the below  
~~~
import test.py
test.test()
~~~
At every ticker, we also display the value of the portfolio which is the sum of the value of fill prices of all option positions, value of underlying and the cash inflow/outflow

# Limitations
The training process and parameters in the paper couldnt be exactly replicated and hence the results give some predictions which can be off. 
This was due to the enormity of the data and since we have options only one underlying, we were also constrained by lack of variety while learning the regressor.  
The training process is conducted only on 2 days worth of data with a final grid loss (as described in paper) close to 2500 after 100 iterations which is not ideal. This can be improved over time.

# Future enhancements
First and foremost is to improve the training results. This will require some time, patience and parameter tuning on the data we have.

The trading strategies can be evolved to include all kinds of exposure, on vol, vega, gamma. These strategies require careful backtesting and cant be fully automated to prevent a code blowout and need to be monitored actively by a human to input live market information. 