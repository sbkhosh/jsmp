#!/usr/bin/python3

import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yaml

from dask import dataframe as dd
from datetime import datetime, timedelta
from dt_help import Helper
from scipy.stats import norm,shapiro,skew,kurtosis
from sqlalchemy import create_engine

import dabl
import datatable as dt

class HistData():
    def __init__(self,input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.new_db_train = self.conf.get('new_db_train')
            
    @Helper.timing
    def process(self):
        # get the data
        HistData.get_data(self)
        
        # compute returns as (P_t-P_{t-1})/P_{t-1} along with log rturns too
        # self.dt_select = self.dt_select.join(pd.DataFrame(self.dt_select[self.ohlc].apply(lambda x: x.pct_change().fillna(0)).values,
        #                           columns=pd.MultiIndex.from_product([['rel_ret'], self.dt_select[self.ohlc].columns]),
        #                           index=self.dt_select.index))
        # self.dt_select = self.dt_select.join(pd.DataFrame(self.dt_select[self.ohlc].apply(lambda x: np.log(x).diff().fillna(0)).values,
        #                           columns=pd.MultiIndex.from_product([['log_ret'], self.dt_select[self.ohlc].columns]),
        #                           index=self.dt_select.index))

        # # print(self.dt_select[self.ohlc][['AAPL','AMZN','NVDA','GOOGL','ADBE','FB','MSFT','TSLA','PYPL']])
        # Helper.beta_calc(self.dt_select)
        # # print(self.dt_select)
        # # self.denoise_detrend = Helper.get_denoise_detrend(self.dt_select,'rel_ret')
        
        # # self.dt_skew = [ Helper.skew_group(self.denoise_detrend,'weekly','denoise_detrend'), Helper.skew_group(self.denoise_detrend,'monthly','denoise_detrend') ]
        # # Helper.skew_plt(self.dt_skew[0],self.dt_skew[1])
        
        # # isolate dataframes used for the clustering step
        # # self.dt_clustering_ret = self.dt_select['rel_ret']
        # # self.dt_clustering_raw = self.dt_select

        # # Helper.plot_cumulative_ret(self.dt_select)
        # # Helper.plot_cumulative_log_ret(self.dt_select)

        # # compute the Intrinsic Mode Functions of the price for
        # # each ticker these will be used in the second strategy
        # # res = []
        # # for el in final_tickers:
        # #     df = self.dt_select[[el]]
        # #     res.append(Helper.get_imfs_hilbert_ts(df,el))
        # # self.df_all_ticks_imfs = pd.concat(res,axis=1)
        
    @Helper.timing
    def get_data(self):
        data_str = ['resps','features_0','features_1','features_2','features_3','others']
        
        if(self.new_db_train):
            train_datatable = dt.fread(self.input_directory + "/csv/train.csv")
            df_train = train_datatable.to_pandas()
            train_datatable = Helper.release_mem_df(train_datatable)
            
            cols = [el for el in df_train.columns if 'feature_' in el]
            len_cols = len(cols)
            
            resps = df_train[[el for el in df_train.columns if 'resp' in el]]
            others = df_train[['date','weight','ts_id']]

            ranges = [slice(0,len_cols//4,1),slice(len_cols//4,len_cols//2,1),slice(len_cols//2,3*len_cols//4,1),slice(3*len_cols//4,len_cols,1)]
            for i,el in enumerate(ranges):
                features = df_train[cols[el]]
                engine_features = create_engine("sqlite:///" + self.output_directory + "/features_"+ str(i)+".db", echo=False)
                features.to_sql(
                    'features_'+str(i).lower(),
                    engine_features,
                    if_exists='replace',
                    index=True,
                )
                features = Helper.release_mem_df(features)

            engine_features_0 = create_engine("sqlite:///" + self.output_directory + "/features_0.db", echo=False)
            engine_features_1 = create_engine("sqlite:///" + self.output_directory + "/features_1.db", echo=False)
            engine_features_2 = create_engine("sqlite:///" + self.output_directory + "/features_2.db", echo=False)
            engine_features_3 = create_engine("sqlite:///" + self.output_directory + "/features_3.db", echo=False)

            df_train = Helper.release_mem_df(features)
            
            engine_resps = create_engine("sqlite:///" + self.output_directory + "/resps.db", echo=False)
            resps.to_sql(
                'resps'.lower(),
                engine_resps,
                if_exists='replace',
                index=True,
            )
            
            engine_others = create_engine("sqlite:///" + self.output_directory + "/others.db", echo=False)
            others.to_sql(
                'others',
                engine_others,
                if_exists='replace',
                index=True,
            )
            
            engines = [engine_resps,engine_features_0,engine_features_1,engine_features_2,engine_features_3,engine_others]
        else:
            engines = [ create_engine("sqlite:///" + self.output_directory + "/"+str(el)+".db", echo=False) for el in data_str ]

        dct_str_eng = dict(zip(data_str,engines))
        all_data = [ pd.read_sql_table(k,con=v) for k,v in dct_str_eng.items() ]
        
        self.df_train = pd.concat(all_data,axis=1).drop(columns=['index'])
        all_data = [ Helper.release_mem_df(el) for el in all_data ]
        
    def get_info_1d(cum_rets):
        end = np.argmax((np.maximum.accumulate(cum_rets)-cum_rets)/np.maximum.accumulate(cum_rets))
        start = np.argmax(cum_rets[:end])
        mdd = np.round(cum_rets[end]-cum_rets[start],2) * 100
        mdd_duration = (cum_rets.index[end]-cum_rets.index[start]).days
        start_date = cum_rets.index[start]
        end_date = cum_rets.index[end]
        cagr = (cum_rets[-1]/cum_rets[1]) ** (252.0 / len(cum_rets)) - 1

        return({'mdd': mdd,'mdd_duration': mdd_duration,
                'start': int(start), 'end': int(end),
                'start_date': start_date , 'end_date': end_date,
                'cum_ret_start': cum_rets[start_date], 'cum_ret_end': cum_rets[end_date],
                'cagr': cagr}) 


        

        

