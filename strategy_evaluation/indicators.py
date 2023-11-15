

import datetime as dt
import os
import io
from io import TextIOWrapper
import re
import numpy as np
import math
import pandas as pd
from util import get_data, plot_data
# import matplotlib.pyplot as plt
# import pdb

def author(): 
  return 'gdutka3'

def getCCI(dates):
   df = pd.DataFrame(index=dates)
   # get High, Low, and adj close
   df['High'] = get_data(["JPM"], dates, addSPY=False, colname="High")
   df['Low'] = get_data(["JPM"], dates, addSPY=False, colname="Low")
   df['Close'] = get_data(["JPM"], dates, addSPY=False, colname="Close")
   # ffill and bfill
   df.fillna(method='ffill', inplace=True)
   df.fillna(method='bfill', inplace=True)

   # get tp values
   df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
   df['sma'] = df['tp'].rolling(14).mean()
   df['mad'] = df['tp'].rolling(14).apply(lambda x: pd.Series(x).mad(),raw=True)
   df['CCI'] = (df['tp'] - df['sma']) / (0.015 * df['mad'])
   return df

def getMomentum(prices):
   momentum = pd.DataFrame(index=prices.index)
   momentum['momentum'] = 0.0
   momentum.iloc[0:14,0] = None
   for i in range(14,len(prices)):
      momentum.iloc[i][0] = prices.iloc[i][0]/prices.iloc[i-14][0] - 1
   return momentum

def getMacd(prices):
   exp1 = prices.ewm(span= 12, adjust= False ).mean()
   exp2 = prices.ewm(span=26, adjust=False).mean()
   macd = exp1 - exp2
   signal = macd.ewm(span=9,adjust=False).mean()
   converge_diverge = macd - signal
   return exp1, exp2, macd, signal, converge_diverge

def getSma(prices, window):
   prices = prices/prices.iloc[0]
   sma = prices.rolling(window).mean()
   ratio = prices / sma
   return sma, ratio

def getBollingerBand(prices):
   sma = prices.rolling(10).mean()
   std = prices.rolling(window=10).std()
   upper_band = sma + (1.5 * std)
   lower_band = sma - (1.5 * std)
   sma = sma/sma.iloc[9]
   bbi = (prices-lower_band)/(upper_band-lower_band)
   return upper_band, lower_band, sma, bbi


# get indicators will return Bollinger bands, price/SMA crossover, RSI

def getIndicators(prices, start_date, end_date):
   # normalize prices
   normed_prices = prices/prices.iloc[0][0]
   
   # create indicators dataframe
   df_indicators = pd.DataFrame(index=prices.index)
   df_indicators['price'] = normed_prices
   
   # 1. get sma and sma/ratio
   sma,smaRatio = getSma(prices, 20)
   df_indicators['sma'] = sma
   df_indicators['price_sma_ratio'] = smaRatio
   # plt.clf()
   #df_indicators['price'].plot(grid=True, linewidth= 1)
   #df_indicators['sma'].plot(grid=True, linewidth= 0.8)
   # plt.xlabel('dates')
   # plt.ylabel('Normalized return')
   # plt.title('price/sma cross')
   # plt.legend(["price","sma","price/sma"])
   # plt.savefig('./images/price-sma-cross.png')
   # plt.clf()
   
   # 2. bollinger bands
   upper_bb, lower_bb, sma, bbp = getBollingerBand(normed_prices)
   df_indicators['sma'] = sma
   df_indicators['upper band'] = upper_bb
   df_indicators['lower band'] = lower_bb
   # much easier to discretize bbp than just bb
   df_indicators['bbp'] = bbp
   # df_indicators['price'].plot(grid=True,label="price").plot(linewidth=1.3)
   # df_indicators['upper band'].plot(grid=True,label="upper bb", linestyle='--', linewidth=1)
   # df_indicators['lower band'].plot(grid=True,label="lower bb", linestyle='--', linewidth=1)
   # df_indicators['sma'].plot(grid=True,label="sma", linewidth=1.2)
   # # plt.xlabel('dates')
   # plt.ylabel('Normalized return')
   # plt.title('15 day bollinger band')
   # plt.legend(["price","upper bb","lower bb", "sma"])
   # plt.savefig('./images/bollingerBands.png')
   # plt.clf()
   
   # 3. momentum
   # fig, axes_1 = plt.subplots(nrows=2)
   # momentum = getMomentum(normed_prices)
   # df_indicators['momentum'] = momentum
   # ax_1 = df_indicators['price'].plot(grid=True,ax=axes_1[0], label='price', linewidth=1, color='orange' )
   # ax_2 = df_indicators['momentum'].plot(grid=True,ax=axes_1[1], label='momentum', linewidth=1, color='black' )
   # ax_1.legend(['price'])
   # ax_1.set_xticks([])
   # ax_2.axhline(linewidth=1, color='r',linestyle='--')
   # ax_2.legend(['momentum','buy-sell line'])
   # ax_1.set_ylabel('price')
   # ax_2.set_xlabel('date')
   # ax_2.set_ylabel('Momentum')
   # ax_1.set_title('Momentum')
   # ax_1.grid(True)
   # plt.savefig('./images/momentum.png')
   # plt.clf()

   # 4. cci
   #import pdb;pdb.set_trace()
   # fig, axes_2 = plt.subplots(nrows=2)
   cci = getCCI(prices.index)#(prices - prices.rolling(window=20).mean() )/(2.5-prices.std())
   df_indicators['cci'] = cci['CCI']
   # chart1 = df_indicators['price'].plot(ax=axes_2[0],grid=True,label='price', linewidth=1, color='blue' )
   # chart2 = cci['CCI'].plot(ax=axes_2[1],grid=True,label='cci', linewidth=1, color='red')
   # chart1.set_xticks([])
   # chart2.set_xlabel('date')
   # chart1.set_title('Commodity Channel Index')
   # chart1.set_ylabel('price')
   # chart2.set_ylabel('cci')
   # chart1.legend(['price'])
   # chart2.legend(['cci'])
   # chart2.axhline(y=100, linewidth=0.8, color='black',linestyle='--')
   # chart2.axhline(y=-100, linewidth=0.8, color='black',linestyle='--')
   # plt.xlabel('dates')
   # plt.ylabel('cci')
   # plt.savefig('./images/cci.png')
   # plt.clf()


   # 5. macd
   # fig, axes = plt.subplots(nrows=2)
   exp1, exp2, macd, signal, converge_diverge =getMacd(prices)
   exp1 = exp1 / exp1.iloc[0][0]
   exp2 = exp2 / exp2.iloc[0][0]
   df_indicators['12 day ema'] = exp1
   df_indicators['26 day ema'] = exp2
   df_indicators['macd'] = macd
   df_indicators['macd_signal'] = signal
   #df_indicators['macd_div_signal'] = df_indicators['macd']/df_indicators['macd_signal']
   # df_indicators['price'].plot(grid=True,ax=axes[0],label='price', linewidth=1 )
   # df_indicators['26 day ema'].plot(grid=True,ax=axes[0],label='26 day ema', linewidth=1)
   # ax = df_indicators['12 day ema'].`plot`(grid=True,ax=axes[0],label='12 day ema', linewidth=1)
   # ax.legend(['price','12 day ema','26 day ema'])
   # ax.set_xticks([])
   # ax.set_ylabel('Price')
   # ax.set_title("MACD")
   # df_indicators['macd'].plot(grid=True,ax=axes[1], label='macd', linewidth=1)
   # ax2 = df_indicators['macd_signal'].plot(grid=True,ax=axes[1], label='macd signal', linewidth=1)
   # ax2.set_xlabel('dates')
   # ax2.legend(['macd','macd_signal'])
   # plt.savefig('./images/macd.png')
   # plt.clf()
   return df_indicators


