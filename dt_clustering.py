#!/usr/bin/python3

import base64
import datetime as dt
import itertools
import joblib
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import re
import scipy
import scipy.cluster.hierarchy as hac
import scipy.stats as stats
import seaborn as sns
import sklearn
import statsmodels.api as sm
import sys, os
import time
import yaml
import warnings

from datetime import datetime, timedelta
from dt_help import Helper
from fastdtw import fastdtw
from joblib import Parallel, delayed
from matplotlib import style
from pandas.plotting import register_matplotlib_converters,scatter_matrix
from pandas.tseries.offsets import BDay
from PyEMD import EMD
from pylab import *
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 

class Clustering():
    def __init__(self,input_directory, output_directory, input_prm_file, data_cluster_ret, data_cluster_raw, data_fundamental):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file
        self.data_cluster_ret = data_cluster_ret
        self.data_cluster_raw = data_cluster_raw
        self.data_fundamental = data_fundamental
        
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
        self.clustering = self.conf.get('clustering')
        self.pca_comps = self.conf.get('clustering')
        self.eps_dbscan = self.conf.get('eps_dbscan')
        self.min_samples_dbscan = self.conf.get('min_samples_dbscan')
        self.metric_dbscan = self.conf.get('metric_dbscan')
        self.method_hac = self.conf.get('method_hac')
        self.metric_hac = self.conf.get('metric_hac')
        self.max_cluster_hac = self.conf.get('max_cluster_hac')

    @staticmethod
    def dtw_ts_fast(ts1,ts2):
        d, path = fastdtw(ts1,ts2)
        return(d)

    @staticmethod
    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = hac.dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return(ddata)
            
    @staticmethod
    def add_distance(ddata, dist_threshold=None, fontsize=8):
        '''
        plot cluster points & distance labels in dendrogram
        ddata: scipy dendrogram output
        dist_threshold: distance threshold where label will be drawn, 
        if None, 1/10 from base leafs will not be labelled to prevent clutter
        fontsize: size of distance labels
        '''
        if dist_threshold==None:
            # add labels except for 1/10 from base leaf nodes
            dist_threshold = max([a for i in ddata['dcoord'] for a in i])/10
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            y = sum(i[1:3])/2
            x = d[1]
            # only label above distance threshold
            if x > dist_threshold:
                plt.plot(x, y, 'o', c=c, markeredgewidth=0)
                plt.annotate(int(x), (x, y), xytext=(15, 3),
                            textcoords='offset points',
                            va='top', ha='center', fontsize=fontsize)

    @staticmethod
    @Helper.timing
    def maxclust_draw_rep(df, method, metric, max_cluster, ts_space=1):
        '''
        draw agglomerative clustering dendrogram based on maximum cluster criteron

        df: dataframe or arrays of timeseries
        method: agglomerative clustering linkage method
        metric: distance metrics
        max_cluster: maximum cluster size to flatten cluster
        ts_space: horizontal space for timeseries graph to be plotted

        gets the dendrogram with timeseries graphs on the side
        '''
        cols = df.T.columns.values

        # define gridspec space
        gs = gridspec.GridSpec(max_cluster,max_cluster)

        # add dendrogram to gridspec
        fig, ax = plt.subplots(figsize=(32,20))
        plt.subplot(gs[:, 0:max_cluster-ts_space])
        plt.xlabel('distance')
        plt.ylabel('cluster')

        # agglomerative clustering
        if(metric != 'dtw'):
            # Z = hac.linkage(df, method=method, metric=metric)
            dm = pdist(df.values,metric=metric)
            Z = hac.linkage(dm,method=method)
        elif(metric == 'dtw'):
            dm = pdist(df,lambda u,v: dtw_ts_fast(u,v))
            Z = hac.linkage(dm,method=method)

        ddata = hac.dendrogram(Z, orientation='left',
                               truncate_mode='lastp',
                               p=max_cluster,
                               show_leaf_counts=True,
                               labels=list(cols),
                               show_contracted=True)

        # add distance labels in dendrogram
        Clustering.add_distance(ddata)

        # get cluster labels
        y = fcluster(Z, max_cluster, criterion='maxclust')
        y = pd.DataFrame(y,columns=['y'])

        # get individual names for each cluster
        df_clst = pd.DataFrame()
        df_clst['index']  = df.index
        df_clst['label']  = y

        # summarize info for output
        dct_sum = {'cluster': [], 'cluster_size': [], 'components': []}
        for i in range(max_cluster):
            elements = df_clst[df_clst['label']==i+1]['index'].tolist()
            size = len(elements)
            dct_sum['cluster'].append(i+1)
            dct_sum['cluster_size'].append(size)
            dct_sum['components'].append('\n'+' | '.join(elements))
        df_sum = pd.DataFrame.from_dict(dct_sum)
        df_sum['components'] = df_sum['components'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]\n", '', x).replace('\n','').replace(' ',''))

        # merge with original dataset
        dx=pd.concat([df.reset_index(drop=True), y],axis=1)

        # add timeseries graphs to gridspec
        dgg = df.copy(deep=True)
        dgg['y']=y.values
        dgg['Dates']=[ str(el) for el in dgg.index ]
        cols_select = [ el for el in list(dgg.columns) if el.__class__.__name__ == 'Timestamp' ]

        df_plots = []
        for el in ddata['ivl']:
            if('(' not in el and ')' not in el):
                df_plots.append(list(dgg[dgg['Dates']==el][cols_select].values[0]))
            elif('(' in el):
                intt = int(str(el).split('(')[1].split(')')[0])
                ts_comps = df_sum[df_sum['cluster_size']==intt]['components'].values
                ts = [ re.sub(r"[^a-zA-Z0-9]\n", '', k).replace('\n','').replace(' ','') for k in ts_comps[0].split('|') ]
                df_plots.append([ list(dgg[dgg['Dates']==el][cols_select].values[0]) for el in ts ])

        for cluster in range(1,max_cluster+1):
            plt.subplot(gs[max_cluster-cluster:max_cluster-cluster+1,max_cluster-ts_space:max_cluster])
            if(np.sum(list(any(isinstance(el, list) for el in df_plots[cluster-1]))) > 0):
                [ plt.plot(el) for el in df_plots[cluster-1] ]
            else:
                plt.plot(df_plots[cluster-1])

        plt.tight_layout()
        plt.savefig('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.pdf')
        df_sum.to_csv('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.csv')
        return(df_sum, Z, ddata, dm)

    @Helper.timing
    def get_clusters_kmeans(self):
        df = self.data_cluster_ret
        prices = np.array(df[df.columns.values]).T
        
        normalizer = Normalizer()
        kmeans = KMeans(n_clusters = 10,max_iter = 1000)
        pipeline = make_pipeline(normalizer,kmeans)
        pipeline.fit(prices)
        labels = pipeline.predict(prices)
        clusters = pd.DataFrame({'labels': labels,
                                 'companies':list(self.companies)}).sort_values(by=['labels'],axis = 0)
        
    @Helper.timing
    def get_clusters_hac(self):
        df = self.data_cluster_ret
        Clustering.maxclust_draw_rep(df.iloc[:,:].T, self.method_hac, self.metric_hac, self.max_cluster_hac, ts_space=1)

    @Helper.timing
    def get_clusters_dbscan_search(self):
        returns = self.data_cluster_ret
        fundamentals = self.data_fundamental

        for comp in range(1,51):
            pca = PCA(n_components=comp)
            pca.fit_transform(returns)

            # data is first reduced to the first 'self.pca_comps' principal components loadings
            # Some additional features are added from the information retrieved from each company
            X = np.hstack(
                (pca.components_.T,
                fundamentals['marketCap'][returns.columns].values[:, np.newaxis])
                )

            for min_sample in range(1,11):
                clf = DBSCAN(eps=0.5, min_samples=min_sample)
                clf.fit(X)
                labels = clf.labels_
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                print(comp,min_sample,n_clusters_)

    @Helper.timing
    def get_clusters_dbscan(self):
        returns = self.data_cluster_ret
        fundamentals = self.data_fundamental

        pca = PCA(n_components=25)
        pca.fit_transform(returns)

        # data is first reduced to the first 'self.pca_comps' principal components loadings
        # Some additional features are added from the information retrieved from each company
        self.X = np.hstack(
            (pca.components_.T,
            fundamentals['marketCap'][returns.columns].values[:, np.newaxis])
            )

        clf = DBSCAN(eps=0.5, min_samples=2)
        clf.fit(self.X)
        self.labels = clf.labels_
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        clustered = clf.labels_

        # as we are looking for pairs in the initial universe of tickers
        # the initial dimensionality of the search is the following
        ticker_count = len(returns.columns)
        print("############################################")
        print("total pairs possible in universe: %d " % (ticker_count*(ticker_count-1)/2))
        print("############################################")

        clustered_ser = pd.Series(index=returns.columns, data=clustered.flatten())
        clustered_ser_all = pd.Series(index=returns.columns, data=clustered.flatten())
        clustered_ser = clustered_ser[clustered_ser != -1]
        self.clustered_ser = clustered_ser

        # a very high value to make sure all elements included in the clusters
        CLUSTER_SIZE_LIMIT = 9999
        counts = clustered_ser.value_counts()
        self.ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]

        print("Clusters formed: %d" % len(self.ticker_count_reduced))
        print("Pairs to evaluate: %d" % (self.ticker_count_reduced*(self.ticker_count_reduced-1)).sum())
        print("############################################")

        # we have found a reduced number of cluster. Now we want to visualize the higher dimension information in 2d.
        # For this purpose, we try T-SNE which is an algorithm for visualizing very high dimension data in 2d
        # Clustering.plot_clusters_dbscan(self)
        # Clustering.plot_ts_clusters_dbscan(self)
        Clustering.plot_clusters_coint(self)
        Clustering.plot_prices_coint(self)
        
    @Helper.timing
    def plot_clusters_dbscan(self):
        X_tsne = TSNE(learning_rate=100, perplexity=25, random_state=253476).fit_transform(self.X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32,20))

        # cumulative return from relative returns definition
        ax1.scatter(
            X_tsne[(self.labels!=-1), 0],
            X_tsne[(self.labels!=-1), 1],
            s=100,
            alpha=0.85,
            c=self.labels[self.labels!=-1],
            cmap=cm.Paired
        )

        # second way of computing from log returns (as a check)
        ax2.barh(
            range(len(self.clustered_ser.value_counts())),
            self.clustered_ser.value_counts()
        )

        plt.xlabel('stocks in cluster')
        plt.ylabel('cluster number');
        # plt.show()

    @Helper.timing
    def plot_ts_clusters_dbscan(self):
        df_prices = self.data_cluster_raw
        
        # get the number of stocks in each cluster        
        counts = self.clustered_ser.value_counts()

        # visualize some clusters
        cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]

        # plot a handful of the smallest clusters
        # prices data is rescaled to the first value of the time series
        for clust in cluster_vis_list:
            tickers = list(self.clustered_ser[self.clustered_ser==clust].index)
            data = df_prices[tickers]/df_prices[tickers].iloc[0]
            data.plot(title='stock time series for cluster %d' % clust)
        # plt.show()

    @Helper.timing
    def plot_clusters_coint(self):
        df_prices = self.data_cluster_raw
        returns = self.data_cluster_ret
        
        # Now that we have clusters of stocks we look at cointegration relationships
        cluster_dict = {}
        for i, which_clust in enumerate(self.ticker_count_reduced.index):
            tickers = self.clustered_ser[self.clustered_ser == which_clust].index
            score_matrix, pvalue_matrix, pairs = Helper.find_cointegrated_pairs(df_prices[tickers])
            cluster_dict[which_clust] = {}
            cluster_dict[which_clust]['score_matrix'] = score_matrix
            cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
            cluster_dict[which_clust]['pairs'] = pairs

        final_pairs = []
        for clust in cluster_dict.keys():
            final_pairs.extend(cluster_dict[clust]['pairs'])
        self.final_pairs = final_pairs
        
        print("we found {} pairs".format(str(len(final_pairs))))
        print("in those pairs, there are {} unique tickers".format(str(len(np.unique(final_pairs)))))

        stocks = np.unique(final_pairs)
        X_df = pd.DataFrame(index=returns.T.index, data=self.X)
        in_pairs_ser = self.clustered_ser.loc[stocks]
        stocks = list(np.unique(final_pairs))
        X_pairs = X_df.loc[stocks]

        X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

        fig, ax = plt.subplots(1, 1, figsize=(32,20))
        
        for pair in final_pairs:
            ticker1 = pair[0]
            loc1 = X_pairs.index.get_loc(pair[0])
            x1, y1 = X_tsne[loc1, :]
            
            ticker2 = pair[1]
            loc2 = X_pairs.index.get_loc(pair[1])
            x2, y2 = X_tsne[loc2, :]

            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');

        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=list(in_pairs_ser.values), cmap=cm.Paired)
        plt.title('T-SNE Visualization of Validated Pairs');
        # plt.show()

    @Helper.timing
    def plot_prices_coint(self):
        df_ratios = pd.DataFrame()
        for el in self.final_pairs:
            ratio = self.data_cluster_raw[el[0]].values/self.data_cluster_raw[el[1]].values
            df_ratios['ratio_'+el[0]+'_'+el[1]] = ratio
        df_ratios.plot(subplots=True,figsize=(32,20))

        axes = plt.gcf().get_axes()
        dct = dict(zip(axes,[df_ratios['ratio_'+el[0]+'_'+el[1]].values for el in self.final_pairs]))
        for k,v in dct.items():
            k.axhline(v.mean())
        # plt.show()
