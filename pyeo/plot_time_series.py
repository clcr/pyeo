# -*- coding: utf-8 -*-
"""
Created on 22 July 2018

@author: Heiko Balzter
"""

###########################################
# plotting time-series data with a confidence interval in Pandas
# written for Python 3.6 on Ubuntu 16
# after https://pandas.pydata.org/pandas-docs/stable/visualization.html
###########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# MAIN
##############################################################################

# define a time series of price data
price = pd.Series(np.random.randn(150).cumsum(), index=pd.date_range('2000-1-1', periods=150, freq='B'))

# calculate a moving average
n = 21 # window size
ma = price.rolling(window = n, center = True).mean()

# and the corresponding standard deviation
mstd = price.rolling(window = n, center = True).std()

# create a figure
plt.figure()

# plot the price data
plt.plot(price.index, price, 'k')

# plot the moving average on top
plt.plot(ma.index, ma, 'b')

# colour in the confidence interval based on 2 times the std
plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)
