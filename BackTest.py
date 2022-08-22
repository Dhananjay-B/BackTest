
# @author: Bhujbal.D.G

# Import required libraries and modules
import pandas as pd
import numpy as np
from datetime import date
from nsepy import get_history
import yfinance as yf
import QuantPack.technicals

# Defining main class
class Run(object):
    def __init__(self):
        self.Indicators = QuantPack.technicals.indicators()
    
    # Fetch data from NSE using NsePy module
    def fetch_data(self, symbol, start, end, index):
        self.symbol = yf.Ticker(symbol)
        return self.symbol.history(start=start, end=end)
    
    # Function for calculating summary of signals generated
    def summary(self, data, account, start, end):
        pd.set_option('display.float_format', '{:.4f}'.format)
        Account = account
        data['Account'] = 0
        data['QTY'] = 0
        data['Gains'] = 0
        data['Balance'] = 0
        data['Profit (%)'] = 0
        
        size = len(data)
        for i in range(0, size):
            data['Account'][i] = account
            data['QTY'][i] = data['Account'][i] / data['Buy'][i]
            data['Gains'][i] = (data['Sell'][i] - data['Buy'][i]) * data['QTY'][i]
            data['Balance'][i] = data['Gains'][i] + data['Account'][i]
            data['Profit (%)'] = data['Gains'] / data['Account'] * 100
            account = data['Balance'][i] 
        
       
        abs_returns = data['Sell'] - data['Buy']
        
        gross_profit = 0
        gross_loss = 0
        for i in range(0, size):
            if abs_returns[i] > 0:
                gross_profit += abs_returns[i]
            else:
                gross_loss += abs_returns[i]
        #print(gross_profit, gross_loss)
        profit_factor = None
        if gross_loss != 0:
            profit_factor = gross_profit / -(gross_loss)
        positive_trades = 0
        negative_trades = 0
        size = len(data)
        for i in range(size):
            if abs_returns[i] > 0:
                positive_trades += 1
            else:
                negative_trades += 1                
        
        start = start
        end = end
        duration = (end - start).days
        years = duration / 365
        total_trades = len(data)
        Equity_Final = data['Balance'].iloc[-1]
        Equity_Peak = data['Balance'].max()
        Returns = (data['Balance'].iloc[-1] - data['Account'].iloc[0]) * 100 / data['Account'].iloc[0]
        Best_Trade = data['Profit (%)'].max()
        Worst_Trade = data['Profit (%)'].min()
        Avg_Trade = data['Profit (%)'].mean()
        Annual_Returns = ((Equity_Final / data['Account'].iloc[0] ) ** (1 / years) - 1) * 100
        
        # Expectancy Ratio
        win_rate = positive_trades * 100 / total_trades
        loss_rate = 100 - win_rate
        avg_profit = gross_profit / positive_trades
        avg_loss = - gross_loss / negative_trades
        reward_risk = avg_profit / avg_loss
        expectancy = reward_risk * win_rate - loss_rate
        
        results = pd.DataFrame([start, end, duration, total_trades, positive_trades, negative_trades, win_rate, Account, Equity_Final,
                                Equity_Peak, Returns, Annual_Returns, Best_Trade, Worst_Trade, Avg_Trade, profit_factor, expectancy],
                               index=['Start', 'End', 'Duration [Days]', 'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate [%]', 'Initial Account', 'Equity Final [Rs]',
                                      'Equity Peak [Rs]', 'Total Return [%]', 'Ann. Returns [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Profit Factor', 'Expectancy [%]'],
                               columns=['Stats'])
        return results
    
    # Simple moving average vs closing prices crossover
    def SMA(self, *, symbol, start, end, index, period = 10, parameter= 'Close', account = 100000, export = False, summary_export = False):
        
        # Fetch data from NSE and calculate SMA from defined module
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.SMA(data = data, period = period, parameter = parameter)
        
        buy_signal = []
        sell_signal = []
        position = False
        
        # Generate signals
        for i in range(0, len(data['Close'])):
            if data['Close'][i] > data['SMA_{}'.format(period)][i]:
                if position == False:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                    position = True
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['Close'][i] < data['SMA_{}'.format(period)][i]:
                if position == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    position = False
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)

        # Add generated signals to dataframe
        data['{}_SMA Buy_signal'.format(period)] = buy_signal
        data['{}_SMA Sell_signal'.format(period)] = sell_signal
        
        # Export file 
        if export == True:
            data.to_csv('{} {}_SMA vs Close BackTest Signals {} to {}.csv'.format(symbol, period, start, end))
            print("File exported by name: '{} {}_SMA vs Close BackTest Signals {} to {}.csv'".format(symbol, period, start, end))
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}_SMA Buy_signal'.format(period)].dropna(how='any'))
        sell = pd.DataFrame(data['{}_SMA Sell_signal'.format(period)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}_SMA vs Close BackTest Results {} to {}.csv'.format(symbol, period, start, end))
            print("File exported by name: '{} {}_SMA vs Close BackTest Results {} to {}.csv'".format(symbol, period, start, end))
        
        return summary

    
    # Exponential moving average vs closing prices crossover
    def EMA(self, *, symbol, start, end, index, period = 10, parameter= 'Close', account = 100000, export = False, summary_export= False):
        
        # Fetch data from NSE and calculate EMA from defined module
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.EMA(data = data, period = period, parameter = parameter)
        
        buy_signal = []
        sell_signal = []
        position = False
        
        # Generate signals
        for i in range(0, len(data['Close'])):
            if data['Close'][i] > data['EMA_{}'.format(period)][i]:
                if position == False:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                    position = True
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['Close'][i] < data['EMA_{}'.format(period)][i]:
                if position == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    position = False
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
        
        # Add generated signals to dataframe
        data['{}_EMA Buy_signal'.format(period)] = buy_signal
        data['{}_EMA Sell_signal'.format(period)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {}_EMA vs Close BackTest Signals {} to {}.csv'.format(symbol, period, start, end))
            print("File exported by name: '{} {}_EMA vs Close BackTest Signals {} to {}.csv'".format(symbol, period, start, end))
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}_EMA Buy_signal'.format(period)].dropna(how='any'))
        sell = pd.DataFrame(data['{}_EMA Sell_signal'.format(period)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}_EMA vs Close BackTest Results {} to {}.csv'.format(symbol, period, start, end))
            print("File exported by name: '{} {}_EMA vs Close BackTest Results {} to {}.csv'".format(symbol, period, start, end))
        
        return summary


    # 2 Simple moving average crossover
    def SMA_Cross(self, *, symbol, start, end, index, long = 21, short = 10, parameter= 'Close', account=100000, export = False, summary_export= False):
        
        # Fetch data from NSE and calculate SMAs from defined module
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.SMA(data = data, period = long, parameter = parameter)
        data = self.Indicators.SMA(data = data, period = short, parameter = parameter)
        
        buy_signal = []
        sell_signal = []
        position = False
        
        # Generate signals
        for i in range(0, len(data['Close'])):
            if data['SMA_{}'.format(short)][i] > data['SMA_{}'.format(long)][i]:
                if position == False:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                    position = True
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['SMA_{}'.format(short)][i] < data['SMA_{}'.format(long)][i]:
                if position == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    position = False
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
         
        # Add generated signals to dataframe
        data['{}/{} SMA Cross Buy_signal'.format(long, short)] = buy_signal
        data['{}/{} SMA Cross Sell_signal'.format(long, short)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {}_SMA vs {}_SMA Cross BackTest Signals {} to {}.csv'.format(symbol, long, short, start, end))
            print("File exported by name: '{} {}_SMA vs {}_SMA Cross BackTest Signals {} to {}.csv'".format(symbol, long, short, start, end))        
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}/{} SMA Cross Buy_signal'.format(long, short)].dropna(how='any'))
        sell = pd.DataFrame(data['{}/{} SMA Cross Sell_signal'.format(long, short)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        summary_data.to_csv('Signals.csv')
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}_SMA vs {}_SMA Cross BackTest Results {} to {}.csv'.format(symbol, long, short, start, end))
            print("File exported by name: '{} {}_SMA vs {}_SMA Cross BackTest Results {} to {}.csv'".format(symbol, long, short, start, end))
        
        return summary
        
    # 2 Exponential moving average crossover
    def EMA_Cross(self, *, symbol, start, end, index, long = 21, short = 10, parameter= 'Close', account = 100000, export = False, summary_export = False):
            
        # Fetch data from NSE and calculate EMAs from defined module
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.EMA(data = data, period = long, parameter = parameter)
        data = self.Indicators.EMA(data = data, period = short, parameter = parameter)
        
        buy_signal = []
        sell_signal = []
        position = False
        
        # Generate signals
        for i in range(0, len(data['Close'])):
            if data['EMA_{}'.format(short)][i] > data['EMA_{}'.format(long)][i]:
                if position == False:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                    position = True
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['EMA_{}'.format(short)][i] < data['EMA_{}'.format(long)][i]:
                if position == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    position = False
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
                
        # Add generated signals to dataframe
        data['{}/{} EMA Cross Buy_signal'.format(long, short)] = buy_signal
        data['{}/{} EMA Cross Sell_signal'.format(long, short)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {}_EMA vs {}_EMA Cross BackTest Signals {} to {}.csv'.format(symbol, long, short, start, end))
            print("File exported by name: '{} {}_EMA vs {}_EMA Cross BackTest Signals {} to {}.csv'".format(symbol, long, short, start, end))        
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}/{} EMA Cross Buy_signal'.format(long, short)].dropna(how='any'))
        sell = pd.DataFrame(data['{}/{} EMA Cross Sell_signal'.format(long, short)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}_EMA vs {}_EMA Cross BackTest Results {} to {}.csv'.format(symbol, long, short, start, end))
            print("File exported by name: '{} {}_EMA vs {}_EMA Cross BackTest Results {} to {}.csv'".format(symbol, long, short, start, end))
        
        return summary
    
    # Function for triple SMA cross strategy
    def Triple_SMA(self, *, symbol, start, end, index, long = 13, middle = 8, short = 5, parameter = 'Close', account = 100000, export = False, summary_export = False):
        
        # Fetch data from NSE and calculate required SMAs
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.SMA(data = data, period = long, parameter = parameter)
        data = self.Indicators.SMA(data = data, period = middle, parameter = parameter)
        data = self.Indicators.SMA(data = data, period = short, parameter = parameter)
        #print(data)
        # Generate buy/sell signals
        buy_signal = []
        sell_signal = []
        position = False
        
        for i in range(0, len(data['Close'])):
            if data['SMA_{}'.format(middle)][i] > data['SMA_{}'.format(long)][i]:
                if data['SMA_{}'.format(short)][i] > data['SMA_{}'.format(middle)][i]:
                    if position == False:
                        buy_signal.append(data['Close'][i])
                        sell_signal.append(np.nan)
                        position = True
                    else:
                        buy_signal.append(np.nan)
                        sell_signal.append(np.nan)
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['SMA_{}'.format(middle)][i] < data['SMA_{}'.format(long)][i]:
                if data['SMA_{}'.format(short)][i] < data['SMA_{}'.format(middle)][i]:
                    if position == True:
                        buy_signal.append(np.nan)
                        sell_signal.append(data['Close'][i])
                        position = False
                    else:
                        buy_signal.append(np.nan)
                        sell_signal.append(np.nan)
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
        
        # Add generated signals to dataframe
        data['{}/{}/{} Triple SMA Cross Buy_signal'.format(long, middle, short)] = buy_signal
        data['{}/{}/{} Triple SMA Cross Sell_signal'.format(long, middle, short)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {}-{}-{} Triple SMA Cross BackTest Signals {} to {}.csv'.format(symbol, long, middle, short, start, end))
            print("File exported by name: '{} {}-{}-{} Triple SMA Cross BackTest Signals {} to {}.csv'".format(symbol, long, middle, short, start, end))
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}/{}/{} Triple SMA Cross Buy_signal'.format(long, middle, short)].dropna(how='any'))
        sell = pd.DataFrame(data['{}/{}/{} Triple SMA Cross Sell_signal'.format(long, middle, short)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}-{}-{} Triple SMA Cross BackTest Results {} to {}.csv'.format(symbol, long, middle, short, start, end))
            print("File exported by name: '{} {}-{}-{} Triple SMA Cross BackTest Results {} to {}.csv'".format(symbol, long, middle, short, start, end))
            
        return summary
    
    # Function for triple EMA cross strategy
    def Triple_EMA(self, *, symbol, start, end, index, long = 13, middle = 8, short = 5, parameter = 'Close', account = 100000, export = False, summary_export = False):
        
        # Fetch data from NSE and calculate required SMAs
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.EMA(data = data, period = long, parameter = parameter)
        data = self.Indicators.EMA(data = data, period = middle, parameter = parameter)
        data = self.Indicators.EMA(data = data, period = short, parameter = parameter)
        #print(data)
        # Generate buy/sell signals
        buy_signal = []
        sell_signal = []
        position = False
        
        for i in range(0, len(data['Close'])):
            if data['EMA_{}'.format(middle)][i] > data['EMA_{}'.format(long)][i]:
                if data['EMA_{}'.format(short)][i] > data['EMA_{}'.format(middle)][i]:
                    if position == False:
                        buy_signal.append(data['Close'][i])
                        sell_signal.append(np.nan)
                        position = True
                    else:
                        buy_signal.append(np.nan)
                        sell_signal.append(np.nan)
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['EMA_{}'.format(middle)][i] < data['EMA_{}'.format(long)][i]:
                if data['EMA_{}'.format(short)][i] < data['EMA_{}'.format(middle)][i]:
                    if position == True:
                        buy_signal.append(np.nan)
                        sell_signal.append(data['Close'][i])
                        position = False
                    else:
                        buy_signal.append(np.nan)
                        sell_signal.append(np.nan)
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
        
        # Add generated signals to dataframe
        data['{}/{}/{} Triple EMA Cross Buy_signal'.format(long, middle, short)] = buy_signal
        data['{}/{}/{} Triple EMA Cross Sell_signal'.format(long, middle, short)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {}-{}-{} Triple EMA Cross BackTest Signals {} to {}.csv'.format(symbol, long, middle, short, start, end))
            print("File exported by name: '{} {}-{}-{} Triple EMA Cross BackTest Signals {} to {}.csv'".format(symbol, long, middle, short, start, end))
        
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}/{}/{} Triple EMA Cross Buy_signal'.format(long, middle, short)].dropna(how='any'))
        sell = pd.DataFrame(data['{}/{}/{} Triple EMA Cross Sell_signal'.format(long, middle, short)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        summary_data.to_csv('Summary.csv')
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export summary file
        if summary_export == True:
            summary.to_csv('{} {}-{}-{} Triple EMA Cross BackTest Results {} to {}.csv'.format(symbol, long, middle, short, start, end))
            print("File exported by name: '{} {}-{}-{} Triple EMA Cross BackTest Results {} to {}.csv'".format(symbol, long, middle, short, start, end))
        
        return summary

    # Moving average convergence/divergence crossover
    def MACD_Cross(self, *, symbol, start, end, index, long = 26, short = 12, signal = 9, parameter= 'Close', account=100000, export = False, summary_export= False):
        
        # Fetch data from NSE and calculate MACD and signal line from defined module
        data = self.fetch_data(symbol = symbol, start = start, end = end, index = index)
        data = self.Indicators.MACD(data = data, long = long, short = short, signal = signal)
        
        # Generate signals
        buy_signal = []
        sell_signal = []
        position = False
        
        for i in range(0, len(data['Close'])):
            if data[f'{signal} Signal'][i] < data[f'{long}-{short} MACD'][i]:
                if position == False:
                    buy_signal.append(data['Close'][i])
                    sell_signal.append(np.nan)
                    position = True
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data[f'{signal} Signal'][i] > data[f'{long}-{short} MACD'][i]:
                if position == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    position = False
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
                
        # Add generated signals to dataframe
        data['{}-{}/{} MACD Buy_signal'.format(long, short, signal)] = buy_signal
        data['{}-{}/{} MACD Sell_signal'.format(long, short, signal)] = sell_signal
        
        # Export file
        if export == True:
            data.to_csv('{} {} {} {} MACD Backtest Signals {} to {}.csv'.format(symbol, long, short, signal, start, end))
            print("File exported by name: '{} {} {} {} MACD Backtest Signals {} to {}.csv'".format(symbol, long, short, signal, start, end))
            
        # Prepare data to pass in to summary function
        buy = pd.DataFrame(data['{}-{}/{} MACD Buy_signal'.format(long, short, signal)].dropna(how='any'))
        sell = pd.DataFrame(data['{}-{}/{} MACD Sell_signal'.format(long, short, signal)].dropna(how='any'))
        if len(buy) > len(sell):
            buy.drop(buy.tail(1).index, inplace= True)
        buy.index = range(len(sell))
        sell.index = range(len(buy))
        summary_data = pd.merge(buy, sell, left_index= True, right_index= True)
        summary_data.columns = ['Buy', 'Sell']
        
        # call summary function
        summary = self.summary(summary_data, account, start, end)
        
        # Export results
        if summary_export == True:
            summary.to_csv('{} {} {} {} MACD Backtest Results {} to {}.csv'.format(symbol, long, short, signal, start, end))
            print("File exported by name: '{} {} {} {} MACD Backtest Results {} to {}.csv'".format(symbol, long, short, signal, start, end))
        
        return summary