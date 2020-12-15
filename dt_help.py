#!/usr/bin/python3

import bs4 as bs
import csv
import gc
import inspect
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go
import random
import re
import requests
import scipy
import seaborn as sns
import time
import yaml

from functools import wraps
from PyEMD import EMD
from pylab import rcParams
from statsmodels.tsa.stattools import coint

class Helper():
    def __init__(self, input_directory, input_prm_file):
        self.input_directory = input_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.format(self.input_directory, self.input_prm_file))

    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.rr_symb1 = self.conf.get('rr_symb1')
        self.rr_symb2 = self.conf.get('rr_symb2')
            
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def get_delim(filename):
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        return(dialect.delimiter)

    @staticmethod
    def get_class_membrs(clss):
        res = inspect.getmembers(clss, lambda a:not(inspect.isroutine(a)))
        return(res)

    @staticmethod
    def check_missing_data(data):
        print(data.isnull().sum().sort_values(ascending=False))

    @staticmethod
    def missing_values_table(dff):
        res = round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False)

        df_res = pd.DataFrame()
        df_res['Missing Value Tickers'] = res.index
        df_res['Total number of samples'] = len(dff)
        df_res['Percentage of missing values'] = res.values
        df_res['Number of missing values'] = len(dff) * df_res['Percentage of missing values'] // 100.0
        df_res = df_res[df_res['Percentage of missing values'] > 0.0]
        return(df_res)

    @staticmethod
    def view_data(data):
        print(data.head())

    @staticmethod
    def plot_cumulative_ret(df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32,20))

        # cumulative return from relative returns definition
        df[[el for el in df.columns if '_rel_ret' in el]].apply(lambda x: (1.0 + x).cumprod()-1.0).plot(ax=ax1)
        ax1.set_ylabel('Cumulative relative returns (1st def)')
        ax1.legend(bbox_to_anchor=(2, 1), loc='upper right', ncol=1)

        # second way of computing from log returns (as a check)
        df[[el for el in df.columns if '_log_ret' in el]].apply(lambda x: (np.exp(x.cumsum()) - 1)).plot(ax=ax2)
        ax2.set_ylabel('Cumulative relative returns (2nd def)')
        ax2.legend().set_visible(False)
        plt.show()

    @staticmethod
    def plot_cumulative_log_ret(df):
        fig, ax = plt.subplots(1, 1, figsize=(32,20))
        
        df[[el for el in df.columns if '_log_ret' in el]].apply(lambda x: x.cumsum()).plot(ax=ax)
        print(df[[el for el in df.columns if '_log_ret' in el]].apply(lambda x: x.cumsum()))
        ax.set_ylabel('Cumulative log returns')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
        plt.show()
        
    @staticmethod
    def get_imfs_hilbert_ts(df,ticker):
        emd = EMD()
        dt = df.index
        ts = df[ticker].values
        IMFs = emd(ts)
        N = IMFs.shape[0]+1
        
        df_imfs = pd.DataFrame()
        for i, el in enumerate(IMFs):
            df_imfs['imf_'+str(i)] = el
        df_imfs['raw'] = df[ticker].values
        df_imfs.columns = ['noise'] + [el for el in df_imfs.columns[1:-2]] + ['trend','raw']
        df_imfs['denoise_detrend'] = df_imfs['raw'] - df_imfs[[el for el in df_imfs.columns if 'imf_' in el]].sum(axis=1)
        df_imfs.insert(0,'Dates',df.index)
        # all_components_emd = df_imfs
        all_feat_emd = df_imfs # [[el for el in df_imfs.columns if 'denoise_' in el]]
        cols = all_feat_emd.columns[1:]
        all_feat_emd.columns = ['Dates'] + [ el + '_' + ticker for el in cols]
        all_feat_emd.set_index('Dates',inplace=True)
        return(all_feat_emd[[el for el in all_feat_emd.columns if 'denoise_' in el]])
        
    @staticmethod
    def find_cointegrated_pairs(data, significance=0.5):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < significance:
                    pairs.append((keys[i], keys[j]))
        return(score_matrix, pvalue_matrix, pairs)

    @staticmethod
    def get_in_out_sample(df,flag,ratio):
        # the data is split into 3 parts:
        # (1) training (2) testing (3) validation
        # the first two are based on in-sample data
        # the last is based on out-of-sample data
        # for this data is taken out completely
        #------------------------------------
        #| 75% of data (1+2) | 25% taken out|              
        #------------------------------------ 
        df_in_sample = df.head(int(len(df)*(ratio)))
        df_out_sample = df.iloc[len(df_in_sample):]

        if(flag=='in'):
            return(df_in_sample)
        elif(flag=='out'):
            return(df_out_sample)

    @staticmethod
    def risk_return(df):
        corr = df.corr()
        # pd.plotting.scatter_matrix(df, diagonal='kde', figsize=(32,20))
        # plt.imshow(corr, cmap='hot', interpolation='none')
        # plt.colorbar()
        # plt.xticks(range(len(corr)), corr.columns)
        # plt.yticks(range(len(corr)), corr.columns)

        plt.figure(figsize=(32,20))
        plt.scatter(df.std(),df.mean())
        plt.xlabel('Risk')
        plt.ylabel('Return')
        for label, x, y in zip(df.columns, df.std(), df.mean()):
            plt.annotate(
                label, 
                xy = (x, y), xytext = (5, -5),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()

    @staticmethod
    def shapiro_test(df):
        p_value = scipy.stats.shapiro(df['log_return'].dropna())[1]
        if p_value <= 0.05:
            return('not_normal')
        else:
            return('normal')

    @staticmethod
    def skew_plt(df0,df1):
        dates_perf0 = df0.index
        dates_perf1 = df1.index

        tickers = df0.columns.get_level_values(1)
        yfmt = mdates.DateFormatter('%Y-%m-%d')

        size = 0.8
        alignment = 0.075
        ratio = 0.5

        fig = plt.figure(figsize=(32,20))
        font = {
            'family': 'serif',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 7,
            }

        ax_0 = fig.add_axes([alignment + 0.00 * size, ratio/4, ratio*size, size])
        ax_1 = fig.add_axes([alignment + 0.60 * size, ratio/4, ratio*size, size])
        
        ax_0.clear()
        ax_1.clear()
        
        ax_0.yaxis.set_major_formatter(yfmt)
        ax_1.yaxis.set_major_formatter(yfmt)

        ax_0.pcolor(df0)
        ax_1.pcolor(df1)

        ax_0.yaxis_date()
        ax_1.yaxis_date()
        
        ax_0.set_yticks(range(len(dates_perf0)))
        ax_0.set_yticklabels(dates_perf0.date,fontdict=font)
        ax_1.set_yticks(range(len(dates_perf1)))
        ax_1.set_yticklabels(dates_perf1.date,fontdict=font)

        ax_0.set_xticks(range(len(tickers)))
        ax_0.set_xticklabels(tickers,rotation=90,fontdict=font)
        ax_1.set_xticks(range(len(tickers)))
        ax_1.set_xticklabels(tickers,rotation=90,fontdict=font)
        
        plt.show()

    @staticmethod
    def skew_group(df,freq,flag):
        if(freq=='hourly'):
            freqs = ['H']
        elif(freq=='weekly'):
            freqs = ['W'] # 'W-MON','W-TUE','W-WED','W-THU','W-FRI','W-SAT','W-SUN'
        elif(freq=='monthly'):
            freqs = ['BM'] # BM
            
        symbs = df[flag].columns
        attrs = [el+'_skew' for el in freqs]
        midx = pd.MultiIndex.from_product([attrs,symbs],names=('Attributes', 'Symbols'))
        
        data_all = [df[flag].groupby(pd.Grouper(freq=el)).skew() for el in freqs]
        dt = pd.concat(data_all,axis=1)
        dt.columns = midx
        return(dt)
        
    @staticmethod
    def get_denoise_detrend(df,ohlc_ret):
        df_denoise_detrend = [ Helper.get_imfs_hilbert_ts(df[ohlc_ret],el) for el in df[ohlc_ret].columns ]

        attrs = ['denoise_detrend']
        symbs = df[ohlc_ret].columns
        midx = pd.MultiIndex.from_product([attrs,symbs],names=('Attributes', 'Symbols'))

        dt = pd.concat(df_denoise_detrend,axis=1)
        dt.columns = midx
        return(dt)

    @staticmethod
    def release_mem_df(df):
        del [[df]]
        gc.collect()
        df = pd.DataFrame()
        return(df)

        
