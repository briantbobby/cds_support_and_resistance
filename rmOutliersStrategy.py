from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import backtrader as bt
import yfinance as yf
import numpy as np
import sys
import pandas as pd
from scipy.signal import savgol_filter
from math import sqrt
import matplotlib.pyplot as plt
import pandas_datareader as web

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 167),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.period = 30

        # Add indicators

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    



    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        series = pd.Series(self.dataclose.get(30,30)).array

        if series.size < 30:
            return

        series.index = np.arange(series.shape[0])

        month_diff = series.shape[0] // 30
        if month_diff == 0:
            month_diff = 1

        #print(month_diff)
        smooth = int(2 * month_diff + 3)
        #print(smooth)
        #print(series.size)
        pts = savgol_filter(series, smooth, 3)

        local_min, local_max = local_min_max(pts)
        
        support = processSR(local_min);
        resistance = processSR(local_max);
        
        #print("Support: " + str(supportPt))
        #print("Resistance: " + str(resistancePt))
        
        diff = resistance - support
        rThreshold = resistance - (diff/6)
        sThreshold = support + (diff/6)
        

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
            
        if (support == -1 or resistance == -1):
            return

        # Check if we are in the market
        if not self.position:
            if (self.dataclose[0] <= sThreshold):
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()

        else:
            if (self.dataclose[0] >= rThreshold):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()
                
                    
def processSR(ptList):
    if (len(ptList) == 0):
        return -1
    
    if (len(ptList) <= 2):
        return sum(ptList)/len(ptList)
    else:
        maxVal = max(ptList)
        ptList.remove(maxVal)
        minVal = min(ptList)
        ptList.remove(minVal)
        return sum(ptList)/len(ptList)
        

def closes(array):
    return array.close
 

def pythag(pt1, pt2):
    a_sq = (pt2[0] - pt1[0]) ** 2
    b_sq = (pt2[1] - pt1[1]) ** 2
    return sqrt(a_sq + b_sq)

def local_min_max(pts):
    local_min = []
    local_max = []
    prev_pts = [(0, pts[0]), (1, pts[1])]
    for i in range(1, len(pts) - 1):
        append_to = ''
        if pts[i-1] > pts[i] < pts[i+1]:
            append_to = 'min'
        elif pts[i-1] < pts[i] > pts[i+1]:
            append_to = 'max'
        if append_to:
            if local_min or local_max:
                prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                curr_distance = pythag(prev_pts[1], (i, pts[i]))
                if curr_distance >= prev_distance:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == 'min':
                        local_min.append(pts[i])
                    else:
                        local_max.append(pts[i])
            else:
                prev_pts[0] = prev_pts[1]
                prev_pts[1] = (i, pts[i])
                if append_to == 'min':
                    local_min.append(pts[i])
                else:
                    local_max.append(pts[i])
    return local_min, local_max

def ticker(name):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Create a Data Feed
    data = bt.feeds.PandasData(
            dataname = yf.download(
                name, 
                datetime.datetime(2010, 1, 1), 
                datetime.datetime(2012, 12, 31)
                )
            )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    #cerebro.plot()



if __name__ == '__main__':
    ticker(sys.argv[1])