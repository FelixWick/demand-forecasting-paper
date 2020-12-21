import sys

import pandas as pd
import numpy as np
import datetime
from dateutil.easter import easter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import nbinom, poisson
from scipy.stats import binned_statistic

from nbpy import binning, pipeline, cyclic_boosting
from nbpy import flags
from nbpy.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from nbpy.cyclic_boosting.plots import plot_analysis
from nbpy.cyclic_boosting.nbinom import CBNBinomC
from nbpy.cyclic_boosting import CBFixedVarianceRegressor
from nbpy.x_transformer import MultiLabelEncoder
from nbpy.cyclic_boosting.price import CBExponential

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj="plots/" + filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


def calculate_factors(est, X):
    factors = {}
    for feature in est.ests[-1][1].features:
        ft = "" if feature.feature_type is None else "_{}".format(
            feature.feature_type)

        factors[
            "_".join(feature.feature_group) + ft
            ] = np.exp(est.ests[-1][1]._pred_feature(X, feature, is_fit=False))
    factors['date'] = X['date']
    factors['yhat_mean'] = X['yhat_mean']
    factors['y'] = X['y']
    df_factors = pd.DataFrame(factors)

    df_factors['item_store'] = df_factors['item_id'] * df_factors['store_id'] * df_factors['item_id_store_id']
    df_factors['price'] = df_factors['promo'] * df_factors['price_ratio'] * df_factors['item_id_promo'] * df_factors['item_id_price_ratio']
    df_factors['dayofweek'] = df_factors['dayofweek'] * df_factors['store_id_dayofweek'] * df_factors['item_id_dayofweek']
    df_factors['weekofmonth'] = df_factors['weekofmonth'] * df_factors['store_id_weekofmonth']
    df_factors['snap'] = df_factors['snap'] * df_factors['snap_dayofweek'] * df_factors['snap_item_id']
    df_factors['seasonality'] = df_factors['td'] * df_factors['dayofyear'] * df_factors['month']
    df_factors['events'] = df_factors['Christmas'] * df_factors['Easter'] * df_factors['event_type_1'] * df_factors['NewYear'] * df_factors['OrthodoxChristmas'] * df_factors['MartinLutherKingDay'] * df_factors['SuperBowl'] * df_factors['ValentinesDay'] * df_factors['PresidentsDay'] * df_factors['StPatricksDay'] * df_factors['OrthodoxEaster'] * df_factors["Mother's day"] * df_factors['MemorialDay'] * df_factors["Father's day"] * df_factors['IndependenceDay'] * df_factors['LaborDay'] * df_factors['ColumbusDay'] * df_factors['Halloween'] * df_factors['VeteransDay'] * df_factors['Thanksgiving'] * df_factors["Cinco De Mayo"] * \
                           df_factors['item_id_event_type_1'] * df_factors['item_id_Christmas'] * df_factors['item_id_Easter'] * df_factors['item_id_NewYear'] * df_factors['item_id_OrthodoxChristmas'] * df_factors['item_id_MartinLutherKingDay'] * df_factors['item_id_SuperBowl'] * df_factors['item_id_ValentinesDay'] * df_factors['item_id_PresidentsDay'] * df_factors['item_id_StPatricksDay'] * df_factors['item_id_OrthodoxEaster'] * df_factors["item_id_Mother's day"] * df_factors['item_id_MemorialDay'] * df_factors["item_id_Father's day"] * df_factors['item_id_IndependenceDay'] * df_factors['item_id_LaborDay'] * df_factors['item_id_ColumbusDay'] * df_factors['item_id_Halloween'] * df_factors['item_id_VeteransDay'] * df_factors['item_id_Thanksgiving'] * df_factors["item_id_Cinco De Mayo"]
    df_factors['yhat_mean'] = df_factors['yhat_mean'] / est.ests[-1][1].global_scale_
    df_factors['y'] = df_factors['y'] / est.ests[-1][1].global_scale_

    df_factors['check'] = df_factors['yhat_mean'] / (df_factors['item_store'] * df_factors['price'] * df_factors['dayofweek'] * df_factors['weekofmonth'] * df_factors['snap'] * df_factors['seasonality'] * df_factors['events'])

    return df_factors


def calculate_factors_width(est, X):
    factors = {}
    for feature in est.ests[-1][1].features:
        ft = "" if feature.feature_type is None else "_{}".format(
            feature.feature_type)

        factors[
            "_".join(feature.feature_group) + ft
            ] = np.exp(est.ests[-1][1]._pred_feature(X, feature, is_fit=False))
    factors['date'] = X['date']
    factors['c'] = X['c']
    df_factors = pd.DataFrame(factors)

    df_factors['item_store'] = df_factors['item_id'] * df_factors['store_id'] * df_factors['item_id_store_id']
    df_factors['price'] = df_factors['promo'] * df_factors['price_ratio']
    df_factors['dayofweek'] = df_factors['dayofweek'] * df_factors['store_id_dayofweek']
    df_factors['seasonality'] = df_factors['td'] * df_factors['dayofyear'] * df_factors['month']
    df_factors['events'] = df_factors['Christmas'] * df_factors['Easter'] * df_factors['event_type_1'] * df_factors['NewYear'] * df_factors['OrthodoxChristmas'] * df_factors['MartinLutherKingDay'] * df_factors['SuperBowl'] * df_factors['ValentinesDay'] * df_factors['PresidentsDay'] * df_factors['StPatricksDay'] * df_factors['OrthodoxEaster'] * df_factors["Mother's day"] * df_factors['MemorialDay'] * df_factors["Father's day"] * df_factors['IndependenceDay'] * df_factors['LaborDay'] * df_factors['ColumbusDay'] * df_factors['Halloween'] * df_factors['VeteransDay'] * df_factors['Thanksgiving'] * df_factors["Cinco De Mayo"] * \
                           df_factors['item_id_event_type_1']

    df_factors['check'] = 1. / (df_factors['c'] * (1 + 1. / (df_factors['item_store'] * df_factors['price'] * df_factors['dayofweek'] * df_factors['weekofmonth'] * df_factors['snap'] * df_factors['seasonality'] * df_factors['events'] * df_factors['yhat_mean_feature'])))

    return df_factors


def plot_factors_ts(df, filename):
    plt.figure()
    ax = plt.axes()
    df.index = df['date']

    df['item_store'].plot(label="item_store")
    df['dayofweek'].plot(label="dayofweek")
    df['weekofmonth'].plot(label="weekofmonth")
    df['snap'].plot(label="snap")
    df['price'].plot(label="price")
    df['seasonality'].plot(label="seasonality")
    df['events'].plot(label="events")
    df['correction_factor'].plot(style='b:', label="correction")
#    df['sales_ewma'].plot(label="sales_ewma")

#    df['check'].plot(label="check")
#    df['y'].plot(label="y")
#    df['yhat_mean'].plot(label="yhat_mean")

    plt.legend(loc=1, fontsize=13)
    plt.ylabel("mean factors", fontsize=13)
    plt.text(-0.1, -0.24, 'b)', fontsize=15, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(filename)


def plot_factors_ts_width(df, filename):
    plt.figure()
    ax = plt.axes()
    df.index = df['date']

    df['item_store'].plot(label="item_store")
    df['dayofweek'].plot(label="dayofweek")
    df['weekofmonth'].plot(label="weekofmonth")
    df['snap'].plot(label="snap")
    df['price'].plot(label="price")
    df['seasonality'].plot(label="seasonality")
    df['events'].plot(label="events")
    df['yhat_mean_feature'].plot(label="yhat_mean")

#    df['check'].plot(label="check")
#    df['c'].plot(label="c")

    plt.legend(loc=1, fontsize=13)
    plt.ylabel("variance factors", fontsize=13)
    plt.text(-0.1, -0.24, 'c)', fontsize=15, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(filename)


def plot_pdf(n, p):
    plt.figure()
    ax = plt.axes()
    x = range(30)
    plt.plot(x, nbinom.pmf(x, n, p))
    plt.xlabel("sales", fontsize=15)
    plt.title('PDF', fontsize=15)
    plt.text(-0.05, -0.15, 'a)', fontsize=15, transform=ax.transAxes)
    plt.tight_layout(rect=(0,0,1,0.99))
    plt.savefig('plots/pdf.pdf')


def plot_cdf(n, p, y):
    plt.figure()
    ax = plt.axes()
    x = range(30)
    plt.plot(x, nbinom.cdf(x, n, p))
    plt.vlines(y, ymin=0, ymax=1, linestyles ="dashed")
    plt.xlabel("sales", fontsize=15)
    plt.title('CDF', fontsize=15)
    plt.text(-0.05, -0.15, 'b)', fontsize=15, transform=ax.transAxes)
    plt.tight_layout(rect=(0,0,1,0.99))
    plt.savefig('plots/cdf.pdf')


def plot_cdf_truth(cdf_truth, suffix, ordering=''):
    plt.figure()
    ax = plt.axes()
    plt.hist(cdf_truth, bins=30)
    if suffix == 'nbinom':
        plt.title('NBD', fontsize=15)
    elif suffix == 'poisson':
        plt.title('Poisson', fontsize=15)
    elif suffix == 'nbinom_larger1':
        plt.title("NBD (>1)", fontsize=15)
    elif suffix == 'poisson_larger1':
        plt.title("Poisson (>1)", fontsize=15)
    else:
        plt.title(suffix, fontsize=20)
    plt.xlabel("CDF values", fontsize=15)
    plt.ylabel("count", fontsize=15)
    plt.text(-0.15, -0.15, ordering, fontsize=15, transform=ax.transAxes)
    plt.hlines(100000./30, xmin=0, xmax=1, linestyles ="dashed")
    plt.tight_layout(rect=(0,0,1,0.99))
    plt.savefig('plots/cdf_truth_' + suffix + '.pdf')


def plot_invquants(X, variable, suffix, continuous=False):
    if suffix == 'nbinom':
        cols = [X['cdf_truth']<=0.1, X['cdf_truth']<=0.3, X['cdf_truth']<=0.5, X['cdf_truth']<=0.7, X['cdf_truth']<=0.9, X['cdf_truth']<=0.97]
    elif suffix == 'poisson':
        cols = [X['cdf_truth_poisson']<=0.1, X['cdf_truth_poisson']<=0.3, X['cdf_truth_poisson']<=0.5, X['cdf_truth_poisson']<=0.7, X['cdf_truth_poisson']<=0.9, X['cdf_truth_poisson']<=0.97]
    if continuous:
        bins = [0., 5., 10., 15. ,20., 25., 30., 35., 40., 45., 50., 55., 60., 70., 80., 100.]
    else:
        bins = 100
    means_result = binned_statistic(X[variable], cols, bins=bins, statistic='mean')
    std_result = binned_statistic(X[variable], cols, bins=bins, statistic='std')
    count_result = binned_statistic(X[variable], cols, bins=bins, statistic='count')
    means10, means30, means50, means70, means90, means97 = means_result.statistic
    std10, std30, std50, std70, std90, std97 = std_result.statistic
    count10, count30, count50, count70, count90, count97 = count_result.statistic
    stdmean10 = std10 / np.sqrt(count10)
    stdmean30 = std30 / np.sqrt(count30)
    stdmean50 = std50 / np.sqrt(count50)
    stdmean70 = std70 / np.sqrt(count70)
    stdmean90 = std90 / np.sqrt(count90)
    stdmean97 = std97 / np.sqrt(count97)

    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    plt.figure()
    ax = plt.axes()
    plt.errorbar(x=bin_centers, y=means10, yerr=stdmean10, linestyle='none', marker='v', c='b')
    plt.errorbar(x=bin_centers, y=means30, yerr=stdmean30, linestyle='none', marker='<', c='g')
    plt.errorbar(x=bin_centers, y=means50, yerr=stdmean50, linestyle='none', marker='o', c='k')
    plt.errorbar(x=bin_centers, y=means70, yerr=stdmean70, linestyle='none', marker='>', c='r')
    plt.errorbar(x=bin_centers, y=means90, yerr=stdmean90, linestyle='none', marker='^', c='y')
    plt.errorbar(x=bin_centers, y=means97, yerr=stdmean97, linestyle='none', marker='s', c='m')
    plt.hlines([0.1, 0.3, 0.5, 0.7, 0.9, 0.97], bin_edges[0], bin_edges[-1], color=['b', 'g', 'k', 'r', 'y', 'm'], linestyle='dashed')
    plt.xlabel(variable, fontsize=15)
    plt.ylabel("quantile", fontsize=14)
    if suffix == 'nbinom':
        plt.title('NBD', fontsize=15)
        plt.text(-0.1, -0.15, 'a)', fontsize=15, transform=ax.transAxes)
    elif suffix == 'poisson':
        plt.title('Poisson', fontsize=15)
        plt.text(-0.1, -0.15, 'b)', fontsize=15, transform=ax.transAxes)
    plt.tight_layout(rect=(0,0,1,0.99))
    plt.savefig('plots/invquant_' + variable + '_' + suffix + '.pdf')


def plot_invquants_examples():
    means10 = [0.05, 0.15, 0.05, 0.15, 0.1]
    means30 = [0.2, 0.4, 0.2, 0.4, 0.3]
    means50 = [0.5, 0.5, 0.45, 0.55, 0.5]
    means70 = [0.8, 0.6, 0.6, 0.8, 0.7]
    means90 = [0.95, 0.85, 0.85, 0.95, 0.9]
    means97 = [0.99, 0.95, 0.95, 0.99, 0.97]

    bins = ['broad', 'narrow', 'over', 'under', 'uniform']

    plt.figure()
    plt.errorbar(x=bins, y=means10, linestyle='none', marker='v', c='b')
    plt.errorbar(x=bins, y=means30, linestyle='none', marker='<', c='g')
    plt.errorbar(x=bins, y=means50, linestyle='none', marker='o', c='k')
    plt.errorbar(x=bins, y=means70, linestyle='none', marker='>', c='r')
    plt.errorbar(x=bins, y=means90, linestyle='none', marker='^', c='y')
    plt.errorbar(x=bins, y=means97, linestyle='none', marker='s', c='m')
    plt.hlines([0.1, 0.3, 0.5, 0.7, 0.9, 0.97], 'broad', 'uniform', color=['b', 'g', 'k', 'r', 'y', 'm'], linestyle='dashed')
    plt.vlines([0.5, 1.5, 2.5, 3.5], 0, 1, linestyle='dotted')
    plt.ylabel("quantile", fontsize=15)
    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/invquant_example.pdf')


def plot_timeseries(df, suffix, title='', textbox=None, errorband=False, split_line=False):
    plt.figure()
    ax = plt.axes()
    df.index = df['date']
    df['y'].plot(style='r', label="sales")
    df['yhat_mean'].plot(style='b-.', label="final prediction")
    if suffix != 'full':
        df['yhat_mean_causal'].plot(style='b:', label="ML prediction")
    if errorband:
        plt.fill_between(df.index, df['yhat_mean'] - np.sqrt(df['yhat_var']), df['yhat_mean'] + np.sqrt(df['yhat_var']), alpha=0.2)
    plt.legend(fontsize=15)
#    plt.title(title)
    plt.ylabel("sum", fontsize=15)
    if split_line:
        plt.vlines('2016-01-01', ymin=0, ymax=80, linestyles="dashed")
    if textbox is not None:
        plt.text(-0.1, -0.22, textbox, fontsize=15, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.pdf'.format(suffix))


def plotting(df, suffix=''):
    df = df[['y', 'yhat_mean', 'yhat_var', 'item_id', 'store_id', 'date', 'correction_factor']]
    df.loc[df['date'] < '2016-01-01', 'correction_factor'] = 1
    df['yhat_mean_causal'] = df['yhat_mean'] / df['correction_factor']
    df.loc[df['date'] < '2016-01-01', 'yhat_mean'] = np.nan

    predictions = df[df['date'] >= '2016-01-01']
    ts_data = predictions.groupby(['date'])[['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix, 'all')

    # predictions_grouped = df.groupby(['store_id'])
    # for name, group in predictions_grouped:
    #     ts_data = group.groupby(['date'])['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal'].sum().reset_index()
    #     plot_timeseries(ts_data, 'store_' + str(name) + suffix)

    # predictions_grouped = df.groupby(['item_id'])
    # for name, group in predictions_grouped:
    #     ts_data = group.groupby(['date'])['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal'].sum().reset_index()
    #     plot_timeseries(ts_data, 'item_' + str(name) + suffix)

    # predictions_grouped = df.groupby(['item_id', 'store_id'])
    # for name, group in predictions_grouped:
    #     ts_data = group.groupby(['date'])['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal'].sum().reset_index()
    #     plot_timeseries(ts_data, 'item_' + str(name) + suffix)

    predictions = df[(df['item_id'] == 16) & (df['store_id'] == 6) & (df['date'] >= '2016-02-01') & (df['date'] < '2016-05-01')]
    ts_data = predictions.groupby(['date'])['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal'].sum().reset_index()
    plot_timeseries(ts_data, 'item_16_store_6' + suffix, textbox='a)', errorband=True)

    predictions = df[(df['item_id'] == 16) & (df['store_id'] == 2) & (df['date'] < '2016-05-01')]
    ts_data = predictions.groupby(['date'])['y', 'yhat_mean', 'yhat_var', 'yhat_mean_causal'].sum().reset_index()
    plot_timeseries(ts_data, 'item_16_store_2_res' + suffix, textbox='a)', errorband=False, split_line=True)


def eval_results(yhat_mean, y):
    mad = np.nanmean(np.abs(y - yhat_mean))
    print('MAD: {}'.format(mad))
    mse = np.nanmean(np.square(y - yhat_mean))
    print('MSE: {}'.format(mse))
    mape = np.nansum(np.abs(y - yhat_mean)) / np.nansum(y)
    print('MAPE: {}'.format(mape))
    smape = 100. * np.nanmean(np.abs(y - yhat_mean) / ((np.abs(y) + np.abs(yhat_mean)) / 2.))
    print('SMAPE: {}'.format(smape))
    md = np.nanmean(y - yhat_mean)
    print('MD: {}'.format(md))

    mean_y = np.nanmean(y)
    print('mean(y): {}'.format(mean_y))


def wasserstein(p, q, n_bins):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # scale by 1/0.5 (0.5 is maximum)
    wasser_distance = 2.0 * np.sum(np.abs(cdf_p - cdf_q)) / len(cdf_p)
    return wasser_distance * n_bins / (n_bins - 1.0)


# def kullback_2(p, q):
#     mask = p > 0  # limit x -> 0 of x log(x) is zero
#     return np.float(np.sum(p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))))
#
#
# def jensen_2(p, q):
#     m = (p + q) / 2.0
#     return (kullback_2(p, m) + kullback_2(q, m)) / 2.0
#
#
# def kullback_e(p, q):
#     mask = p > 0  # limit x -> 0 of x log(x) is zero
#     return np.float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
#
#
# def jensen_e(p, q):
#     m = (p + q) / 2.0
#     return (kullback_e(p, m) + kullback_e(q, m)) / 2.0


def cdf_accuracy(cdf_truth, metric='wasserstein'):
    counts = cdf_truth.value_counts(bins=100)
    n_cdf_bins = len(counts)
    pmf = counts / np.sum(counts)  # relative frequencies for each bin
    unif = np.full_like(pmf, 1.0 / len(pmf))  # uniform distribution
    if metric == 'wasserstein':
        divergence = wasserstein(unif, pmf, n_cdf_bins)
    # elif metric == 'kullback_2':
    #     divergence = kullback_2(unif, pmf)
    # elif metric == 'kullback_e':
    #     divergence = kullback_e(unif, pmf)
    # elif metric == 'jensen_2':
    #     divergence = jensen_2(unif, pmf)
    # elif metric == 'jensen_e':
    #     divergence = jensen_e(unif, pmf)
    cdf_acc = np.float(1.0 - np.clip(divergence, a_min=None, a_max=1.0))
    print('cdf accuracy: {}'.format(cdf_acc))


def set_td_weights_cb(df):
    df['td'] = pd.to_timedelta(df['date']).dt.days
    df['td'] = df['td'] - df['td'].min()
    decay_const = 365. # number of days after the weight is down to 0.5
    const = -np.log(0.5) / decay_const
    df['weights_cb'] = np.exp((df['td'] - df['td'].max()) * const)
    return df


def cut_zeros(df, target='sales', trailing=0):
    first_nonzero = df[target].to_numpy().nonzero()[0][0]
    last_nonzero = df[target].to_numpy().nonzero()[0][-1]
    upper_cut = last_nonzero + trailing
    df.reset_index(drop=True, inplace=True)
    return df.truncate(before=first_nonzero, after=upper_cut)


def cut_zero_blocks(df, target='sales', blocksize=14):
    last_i = 0
    gap_list = []
    df_block = df.reset_index(drop=True)
    for i in  df_block[target].to_numpy().nonzero()[0]:
        gap = i - last_i
        if gap > blocksize:
            gap_list.append(np.arange(last_i+1, i))
        last_i = i
    gap_list = [item for sublist in gap_list for item in sublist]
    df_block.drop(gap_list, inplace=True)
    df_block.reset_index(drop=True, inplace=True)
    return df_block


def get_events(df):
    for event in ['Christmas', 'Easter']:
        for event_date in df['date'][df['event_name_1'] == event].unique():
            for event_days in range(-7, 3):
                df.loc[df['date'] == pd.to_datetime(event_date).date() + datetime.timedelta(days=event_days), event] = event_days

    for event in ['NewYear',
                  'OrthodoxChristmas',
                  'MartinLutherKingDay',
                  'SuperBowl',
                  'ValentinesDay',
                  'PresidentsDay',
                  'StPatricksDay',
                  'OrthodoxEaster',
                  "Mother's day",
                  'MemorialDay',
                  "Father's day",
                  'IndependenceDay',
                  'LaborDay',
                  'ColumbusDay',
                  'Halloween',
                  'VeteransDay',
                  'Thanksgiving',
                  "Cinco De Mayo",
                  ]:
        for event_date in df['date'][df['event_name_1'] == event].unique():
            for event_days in range(-3, 2):
                df.loc[df['date'] == pd.to_datetime(event_date).date() + datetime.timedelta(days=event_days), event] = event_days

    return df


def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = set_td_weights_cb(df)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['weekofmonth'] = np.ceil(df['date'].dt.day / 7)

    df['snap_CA'] = df.loc[df['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4']), 'snap_CA']
    df['snap_TX'] = df.loc[df['store_id'].isin(['TX_1', 'TX_2', 'TX_3']), 'snap_TX']
    df['snap_WI'] = df.loc[df['store_id'].isin(['WI_1', 'WI_2', 'WI_3']), 'snap_WI']
    df['snap'] = df[['snap_CA', 'snap_TX', 'snap_WI']].sum(axis=1)

    df['price_ratio'] = df['sell_price'] / df['list_price']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)
    df.loc[df['price_ratio'] == 1., 'price_ratio'] = np.nan

    df['promo'] = 0
    df.loc[df['price_ratio'] < 1., 'promo'] = 1

    # cut out leading and trailing zero sales
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.groupby(['item_id', 'store_id']).apply(cut_zeros)
    # cut out zero blocks
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.groupby(['item_id', 'store_id']).apply(cut_zero_blocks)
    df.reset_index(drop=True, inplace=True)

    df = get_events(df)

    encoding_cols = ['store_id', 'item_id', 'event_type_1']
    label_encoder = MultiLabelEncoder(selected_columns=encoding_cols, unknowns_as_missing=True)
    df = label_encoder.fit_transform(df)

#    df['sales_ewma_dow'] = emov_feature(df, ['store_id', 'item_id', 'dayofweek'], 0.05, 1)
#    df['sales_ewma'] = emov_feature(df, ['store_id', 'item_id'], 0.25, 2)
#    df.loc[df['sales_ewma_dow'] < 0.1, 'sales_ewma'] = np.nan
#    df.loc[df['sales_ewma'] < 0.1, 'sales_ewma'] = np.nan

    y = np.asarray(df['sales'])
    X = df.drop(columns='sales')
    return X, y


def feature_properties():
    fp = {}
    fp['item_id'] = flags.IS_UNORDERED
    fp['store_id'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['month'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['weekofmonth'] = flags.IS_ORDERED
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['price_ratio'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['snap'] = flags.IS_UNORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['event_type_1'] = flags.IS_UNORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['promo'] = flags.IS_ORDERED
    fp['Christmas'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Easter'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['NewYear'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['OrthodoxChristmas'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['MartinLutherKingDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['SuperBowl'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['ValentinesDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['PresidentsDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['StPatricksDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['OrthodoxEaster'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp["Mother's day"] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['MemorialDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp["Father's day"] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['IndependenceDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['LaborDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['ColumbusDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Halloween'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['VeteransDay'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Thanksgiving'] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp["Cinco De Mayo"] = flags.IS_ORDERED | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['sales_ewma'] = flags.IS_CONTINUOUS | flags.HAS_MISSING
    fp['sales_ewma_dow'] = flags.IS_CONTINUOUS | flags.HAS_MISSING
    return fp


def cb_mean_model():
    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                         }

    features = [
#        'sales_ewma',
#        'sales_ewma_dow',
        'td',
        'dayofweek',
        'store_id',
        'item_id',
        'promo',
        'price_ratio',
        'dayofyear',
        'month',
        'snap',
        'Christmas',
        'Easter',
        'event_type_1',
        'weekofmonth',
        'NewYear',
        'OrthodoxChristmas',
        'MartinLutherKingDay',
        'SuperBowl',
        'ValentinesDay',
        'PresidentsDay',
        'StPatricksDay',
        'OrthodoxEaster',
        "Mother's day",
        'MemorialDay',
        "Father's day",
        'IndependenceDay',
        'LaborDay',
        'ColumbusDay',
        'Halloween',
        'VeteransDay',
        'Thanksgiving',
        "Cinco De Mayo",
        ('item_id', 'store_id'),
        ('item_id', 'promo'),
        ('item_id', 'price_ratio'),
        ('store_id', 'dayofweek'),
        ('item_id', 'dayofweek'),
        ('snap', 'dayofweek'),
        ('snap', 'item_id'),
        ('store_id', 'weekofmonth'),
        ('item_id', 'event_type_1'),
        ('item_id', 'Christmas'),
        ('item_id', 'Easter'),
        ('item_id', 'NewYear'),
        ('item_id', 'OrthodoxChristmas'),
        ('item_id', 'MartinLutherKingDay'),
        ('item_id', 'SuperBowl'),
        ('item_id', 'ValentinesDay'),
        ('item_id', 'PresidentsDay'),
        ('item_id', 'StPatricksDay'),
        ('item_id', 'OrthodoxEaster'),
        ('item_id', "Mother's day"),
        ('item_id', 'MemorialDay'),
        ('item_id', "Father's day"),
        ('item_id', 'IndependenceDay'),
        ('item_id', 'LaborDay'),
        ('item_id', 'ColumbusDay'),
        ('item_id', 'Halloween'),
        ('item_id', 'VeteransDay'),
        ('item_id', 'Thanksgiving'),
        ('item_id', "Cinco De Mayo"),
    ]

    plobs = [cyclic_boosting.observers.PlottingObserver(iteration=-1)]

    est = CBFixedVarianceRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        weight_column='weights_cb',
        maximal_iterations=50,
        smoother_choice=cyclic_boosting.common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = pipeline.Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def cb_width_model():
    fp = feature_properties()
    fp['yhat_mean_feature'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED

    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                          ('yhat_mean_feature',): IsotonicRegressor(increasing=False),
                         }

    features = [
        'yhat_mean_feature',
        'td',
        'dayofweek',
        'store_id',
        'item_id',
        'promo',
        'price_ratio',
        'dayofyear',
        'month',
        'snap',
        'Christmas',
        'Easter',
        'event_type_1',
        'weekofmonth',
        'NewYear',
        'OrthodoxChristmas',
        'MartinLutherKingDay',
        'SuperBowl',
        'ValentinesDay',
        'PresidentsDay',
        'StPatricksDay',
        'OrthodoxEaster',
        "Mother's day",
        'MemorialDay',
        "Father's day",
        'IndependenceDay',
        'LaborDay',
        'ColumbusDay',
        'Halloween',
        'VeteransDay',
        'Thanksgiving',
        "Cinco De Mayo",
        ('item_id', 'store_id'),
        # ('item_id', 'promo'),
        # ('item_id', 'price_ratio'),
        ('store_id', 'dayofweek'),
        # ('item_id', 'dayofweek'),
        # ('snap', 'dayofweek'),
        # ('snap', 'item_id'),
        # ('store_id', 'weekofmonth'),
        ('item_id', 'event_type_1'),
        # ('item_id', 'Christmas'),
        # ('item_id', 'Easter'),
        # ('item_id', 'NewYear'),
        # ('item_id', 'OrthodoxChristmas'),
        # ('item_id', 'MartinLutherKingDay'),
        # ('item_id', 'SuperBowl'),
        # ('item_id', 'ValentinesDay'),
        # ('item_id', 'PresidentsDay'),
        # ('item_id', 'StPatricksDay'),
        # ('item_id', 'OrthodoxEaster'),
        # ('item_id', "Mother's day"),
        # ('item_id', 'MemorialDay'),
        # ('item_id', "Father's day"),
        # ('item_id', 'IndependenceDay'),
        # ('item_id', 'LaborDay'),
        # ('item_id', 'ColumbusDay'),
        # ('item_id', 'Halloween'),
        # ('item_id', 'VeteransDay'),
        # ('item_id', 'Thanksgiving'),
        # ('item_id', "Cinco De Mayo"),
    ]

    plobs = [cyclic_boosting.observers.PlottingObserver(iteration=-1)]

    est = CBNBinomC(
        mean_prediction_column='yhat_mean',
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        weight_column='weights_cb',
        maximal_iterations=50,
        smoother_choice=cyclic_boosting.common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = pipeline.Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def emov_correction(X, group_cols, date_col, horizon=None):
    X.sort_values([date_col], inplace=True)
    X_grouped = X.groupby(group_cols)
    alpha = 0.15

    if horizon is None:
        truth_emov = X_grouped['y'].apply(lambda x: x.ewm(alpha=alpha, ignore_na=True).mean())
        pred_emov = X_grouped['yhat_mean'].apply(lambda x: x.ewm(alpha=alpha, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        X = X[X[date_col] == X[date_col].max()]
        return X[group_cols + ['correction_factor']]
    else:
        truth_emov = X_grouped['y'].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        pred_emov = X_grouped['yhat_mean'].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        return X


def mean_fit(X, y, split_date):
    ml_est_mean = cb_mean_model()
    ml_est_mean.fit(X[X['date'] < split_date].copy(), y[X['date'] < split_date])

    plot_CB('analysis_CB_mean_iterlast', [ml_est_mean.ests[-1][1].observers[-1]], ml_est_mean.ests[-2][1])

    del X
    return ml_est_mean


def mean_predict(X, y, ml_est_mean):
    yhat_mean = ml_est_mean.predict(X)
    X['y'] = y
    X['yhat_mean'] = yhat_mean
    X = X[(X['date'] >= '2016-02-01') & (X['date'] < '2016-05-01') & (X['item_id'] == 16) & (X['store_id'] == 6)]
    X = calculate_factors(ml_est_mean, X)

    return yhat_mean, X


def width_fit(X, split_date):
    ml_est_width = cb_width_model()
    ml_est_width.fit(X[X['date'] < split_date].copy(), np.asarray(X['y'])[X['date'] < split_date])

    plot_CB('analysis_CB_width_iterlast', [ml_est_width.ests[-1][1].observers[-1]], ml_est_width.ests[-2][1])

    del X
    return ml_est_width


def width_predict(X, ml_est_width):
    c = ml_est_width.predict(X)
#    c[X['yhat_mean'] < 1.] = 1
    variance = X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']

    X['c'] = c
    X = X[(X['date'] >= '2016-02-01') & (X['date'] < '2016-05-01') & (X['item_id'] == 16) & (X['store_id'] == 6)]
    X = calculate_factors_width(ml_est_width, X)
    plot_factors_ts_width(X, 'plots/factors_ts_width.pdf')

    del X
    return variance, c


def transform_nbinom(mean, var):
    p = np.minimum(np.where(var > 0, mean / var, 1 - 1e-8), 1 - 1e-8)
    n = np.where(var > 0, mean * p / (1 - p), 1)
    return n, p


def random_from_cdf_interval(X, mode='nbinom'):
    if mode == 'nbinom':
        cdf_high = nbinom.cdf(X['y'], X['n'], X['p'])
        cdf_low = nbinom.cdf(np.where(X['y'] >= 1., X['y'] - 1., 0.), X['n'], X['p'])
    elif mode == 'poisson':
        cdf_high = poisson.cdf(X['y'], X['yhat_mean'])
        cdf_low = poisson.cdf(np.where(X['y'] >= 1., X['y'] - 1., 0.), X['yhat_mean'])
    cdf_low = np.where(X['y'] == 0., 0., cdf_low)
    return cdf_low + np.random.uniform(0, 1, len(cdf_high)) * (cdf_high - cdf_low)


def cdf_truth(X):
    X['n'], X['p'] = transform_nbinom(X['yhat_mean'], X['yhat_var'])
    # item_id: FOODS_3_516, store_id: TX_3
    # X[(X['date'] == '2016-05-06') & (X['item_id'] == 16) & (X['store_id'] == 6)]
    plot_pdf(X['n'][146295], X['p'][146295])
    plot_cdf(X['n'][146295], X['p'][146295], X['y'][146295])
    X['cdf_truth'] = random_from_cdf_interval(X, mode='nbinom')
    X['cdf_truth_poisson'] = random_from_cdf_interval(X, mode='poisson')
    return X


def emov_feature(X, group_cols, alpha, horizon):
    X.sort_values(['date'], inplace=True)
    X_grouped = X.groupby(group_cols)
    return X_grouped['sales'].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())


def main(args):
    # example plots
#    plot_cdf_truth(pd.Series(np.random.rand(100000)), 'uniform', "a)")
    plot_cdf_truth(pd.Series(np.random.triangular(0,0,1,100000)), 'over', "c)")
    plot_cdf_truth(pd.Series(np.random.triangular(0,1,1,100000)), 'under', "d)")
    plot_cdf_truth(pd.Series(np.concatenate((np.random.triangular(0,0.5,0.5,50000),np.random.triangular(0.5,0.5,1,50000)))), 'broad', "a)")
    plot_cdf_truth(pd.Series(np.concatenate((np.random.triangular(0,0,0.5,50000),np.random.triangular(0.5,1,1,50000)))), 'narrow', "b)")
    plot_invquants_examples()

    df = pd.read_csv('data/M5_FOODS_3_5_2013.csv')

    X, y = prepare_data(df)

    split_date = '2016-01-01'

    ml_est_mean = mean_fit(X.copy(), y, split_date)

    X['yhat_mean'], df_factors = mean_predict(X.copy(), y, ml_est_mean)

    X['y'] = y
    horizon = 2
    if horizon is None:
        df_correction = emov_correction(X[X['date'] < split_date], ['store_id', 'item_id'], 'date')
        df_correction.reset_index(drop=True, inplace=True)
        X = X.merge(df_correction, on=['store_id', 'item_id'], how='left')
    else:
        X = emov_correction(X, ['store_id', 'item_id'], 'date', horizon)
    mask = (X['date'] >= split_date) & (pd.notna(X['correction_factor']))
    X['yhat_mean'][mask] = X['correction_factor'][mask] * X['yhat_mean'][mask]

    df_factors['correction_factor'] = X.loc[(X['date'] >= '2016-02-01') & (X['date'] < '2016-05-01') & (X['item_id'] == 16) & (X['store_id'] == 6), 'correction_factor']
    plot_factors_ts(df_factors, 'plots/factors_ts.pdf')

    X['yhat_mean_feature'] = X['yhat_mean']

    ml_est_width = width_fit(X.copy(), split_date)

    X['yhat_var'], X['c'] = width_predict(X.copy(), ml_est_width)

    np.random.seed(42)
    X = cdf_truth(X)

    mask = X['date'] >= split_date
    cdf_accuracy(X['cdf_truth'][mask], 'wasserstein')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'wasserstein')
    # cdf_accuracy(X['cdf_truth'][mask], 'kullback_2')
    # cdf_accuracy(X['cdf_truth_poisson'][mask], 'kullback_2')
    # cdf_accuracy(X['cdf_truth'][mask], 'kullback_e')
    # cdf_accuracy(X['cdf_truth_poisson'][mask], 'kullback_e')
    # cdf_accuracy(X['cdf_truth'][mask], 'jensen_2')
    # cdf_accuracy(X['cdf_truth_poisson'][mask], 'jensen_2')
    # cdf_accuracy(X['cdf_truth'][mask], 'jensen_e')
    # cdf_accuracy(X['cdf_truth_poisson'][mask], 'jensen_e')

    plot_cdf_truth(X['cdf_truth'][mask], 'nbinom', "a)")
    plot_cdf_truth(X['cdf_truth_poisson'][mask], 'poisson', "b)")
    plot_cdf_truth(X['cdf_truth'][mask & (X['yhat_mean'] > 1.)], 'nbinom_larger1', "c)")
    plot_cdf_truth(X['cdf_truth_poisson'][mask & (X['yhat_mean'] > 1.)], 'poisson_larger1', "d)")

    plot_invquants(X[mask], 'yhat_mean', 'nbinom', continuous=True)
    plot_invquants(X[mask], 'store_id', 'nbinom')
    plot_invquants(X[mask], 'dayofweek', 'nbinom')
    plot_invquants(X[mask], 'yhat_mean', 'poisson', continuous=True)
    plot_invquants(X[mask], 'store_id', 'poisson')
    plot_invquants(X[mask], 'dayofweek', 'poisson')

    X['yhat_mean'] = np.round(X['yhat_mean'], 2)
    X['yhat_var'] = np.round(X['yhat_var'], 2)
    X[['item_id', 'store_id', 'date', 'y', 'yhat_mean', 'yhat_var']][mask].to_csv('forecasts_2016.csv', index=False)

    plotting(X[X['date'] >= '2015-11-01'])
#    plotting(X)

    eval_results(X['yhat_mean'][mask], X['y'][mask])

#    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
