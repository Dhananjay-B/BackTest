# BackTest
Python package for backtesting trading strategies 

# Structure

This package contains two modules. Module named 'technicals' defines all the necessary technical indicators which ae used in 'BackTest' module. BackTest module performs
all heavy lifting like accessing data from Yahoo Finance, generating buy/sell signals as per strategy and then calculating summary of signals.

# Exporting Data Results

By default it returns data in form of pandas dataframe and one can export result files as well by passing 'export = True' parameter for exporting signals data and 'summary_export = True' for results data

# Functionalities
There are commonly used indicators defined in 'technicals' module based on which strategies are there in 'BackTest' module. One can add their own technical indicators in
module and hence can create new strategy as per their need.

# Available Indicators
- Simple moving average
- Exponential moving average
- Bollinger band
- Relative strength index
- Moving average convergence divergence

# Available Strategies
- Simple moving average 
- Exponential moving average
- 2 Simple moving average cross
- 2 Exponential moving average
- 3 Simple moving average
- 3 Exponential moving average
- Moving average convergence divergence cross
