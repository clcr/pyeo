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

##############################################################################
# MAIN
##############################################################################

price = pd.Series(np.random.randn(150).cumsum(), index=pd.date_range('2000-1-1', periods=150, freq='B'))

ma = price.rolling(20).mean()

mstd = price.rolling(20).std()

plt.figure()

plt.plot(price.index, price, 'k')

plt.plot(ma.index, ma, 'b')

plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)
