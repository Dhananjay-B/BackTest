# BackTest
Python package for backtesting trading strategies 

This package contains two modules. Module named 'technicals' defines all the necessary technical indicators which ae used in 'BackTest' module. BackTest module performs
all heavy lifting like accessing data from Yahoo Finance, generating buy/sell signals as per strategy and then calculating summary of signals. It returns data in form of
pandas dataframe and one can export result files as well by passing 'export = True' parameter for exporting signals data and 'summary_export = True' for results data
