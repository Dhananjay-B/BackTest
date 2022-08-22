
# @author: Bhujbal.D.G

# Import required modules and libraries
import pandas as pd
import numpy as np
from datetime import date
import math

# Defining class to put all indicator functions inside
class indicators(object):
    
    # Simple moving average function. Returns dataframe passed in with added column of SMA 
    def SMA(self, *, data, period= 10, parameter = 'Close', export = False):
        data['SMA_{}'.format(period)] = data[parameter].rolling(window = period).mean()
        
        # Export file with added data if export is set to True
        if export == True:
            data.to_csv('results.csv')
            print("Data exported to current working directory as 'results.csv'")
            
        return data
    
    # Exponential moving average function. Returns dataframe passed in with added column of EMA
    def EMA(self, *, data, period= 10, parameter = 'Close', export = False):
        data['EMA_{}'.format(period)] = data[parameter].ewm(span=period, min_periods=period).mean()
        
        # Export file with added data if export is set to True
        if export == True:
            data.to_csv('results.csv')
            print("Data exported to current working directory as 'results.csv'")
            
        return data
    
    # Bollinger band function. Returns dataframe passed in with added columns of middle, upper and lower bounds of bollinger band
    def BollingerBand(self, *, data, SMA_Period = 20, parameter= 'Close', Stdev = 2, export = False):
        data['BB_SMA{}'.format(SMA_Period)] = data[parameter].rolling(window = SMA_Period).mean()
        self.BB_STDEV = data[parameter].rolling(window= SMA_Period).std()
        data['BB_UP{}'.format(SMA_Period)] = data['BB_SMA{}'.format(SMA_Period)] + Stdev * self.BB_STDEV
        data['BB_DOWN{}'.format(SMA_Period)] = data['BB_SMA{}'.format(SMA_Period)] - Stdev * self.BB_STDEV
        
        # Export file with added data if export is set to True
        if export == True:
            data.to_csv('results.csv')
            print("Data exported to current working directory as 'results.csv'")
            
        return data
     
    # RSI indicator function. Calculates and add RSI values based on EMA    
    def RSI(self, *, data, period = 14, export = True):
        self.delta = data['Close'].diff()
        self.gains = self.delta.clip(lower = 0)
        self.loss =  -1 * self.delta.clip(upper = 0)
        self.gain_ma = self.gains.ewm(com = period - 1, adjust=True, min_periods = period).mean()
        self.loss_ma = self.loss.ewm(com = period - 1, adjust=True, min_periods = period).mean()
        self.RSI = self.gain_ma / self.loss_ma
        self.RSI = 100 - (100 / (1 + self.RSI))
        data['RSI_{}'.format(period)] = self.RSI
        
        # Export file with added data if export is set to True
        if export == True:
            data.to_csv('results.csv')
            print("Data exported to current working directory as 'results.csv'")
            
        return data
    
    # MACD indicator function. Calculates and add respective fast and slow EMAs and signal line values
    def MACD(self, data, long = 26, short = 12, signal = 9, export = False):
        long_ema = data['Close'].ewm(span=long, min_periods=long).mean()
        short_ema = data['Close'].ewm(span=short, min_periods=short).mean()
        
        data[f'{long}-{short} MACD'] = short_ema - long_ema
        data[f'{signal} Signal'] = data[f'{long}-{short} MACD'].ewm(span= signal, min_periods= signal).mean()
        
        # Export file with added data if export is set to True
        if export == True:
            data.to_csv('results.csv')
            print("Data exported to current working directory as 'results.csv'")
        return data    
        
        