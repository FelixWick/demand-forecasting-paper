import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import nbinom
from scipy.stats import binned_statistic

from nbpy import binning, pipeline, cyclic_boosting
from nbpy import flags
from nbpy.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from nbpy.cyclic_boosting.plots import plot_analysis
from nbpy.cyclic_boosting.nbinom import CBNBinomC

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj="plots/" + filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


def plot_factors(est, X, samples):
    factors = {}
    for feature in est.ests[-1][1].features:
        ft = "" if feature.feature_type is None else "_{}".format(
            feature.feature_type)

        factors[
            "_".join(feature.feature_group) + ft
            ] = np.exp(est.ests[-1][1]._pred_feature(X, feature, is_fit=False)) - 1.
    df_factors = pd.DataFrame(factors)

    ax = df_factors.loc[samples].plot(kind='barh', title="factors", legend=True, fontsize=15)
    fig = ax.get_figure()
    ax.axvline(0, color="gray")
    ax.set_yticks([])

    def setfactorformat(x, pos):
        return '%1.2f' % (x + 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(setfactorformat))

    fig.savefig('plots/factors.pdf')


def plot_cdf(n, p):
    plt.figure()
    x = np.linspace(0.0, 100.0, num=100)
    plt.plot(x, nbinom.cdf(x, n, p))
    plt.tight_layout()
    plt.savefig('plots/cdf.pdf')


def plot_cdf_truth(cdf_truth):
    plt.figure()
    plt.hist(cdf_truth, bins=100)
    plt.tight_layout()
    plt.savefig('plots/cdf_truth.pdf')


def plot_invquants(X, variable):
    means_result = binned_statistic(X[variable], [X['cdf_truth']<=0.1, X['cdf_truth']<=0.3, X['cdf_truth']<=0.5, X['cdf_truth']<=0.7, X['cdf_truth']<=0.9, X['cdf_truth']<=0.97], bins=100, statistic='mean')
    means10, means30, means50, means70, means90, means97 = means_result.statistic

    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    plt.figure()
    plt.errorbar(x=bin_centers, y=means10, linestyle='none', marker='v')
    plt.errorbar(x=bin_centers, y=means30, linestyle='none', marker='<')
    plt.errorbar(x=bin_centers, y=means50, linestyle='none', marker='.')
    plt.errorbar(x=bin_centers, y=means70, linestyle='none', marker='>')
    plt.errorbar(x=bin_centers, y=means90, linestyle='none', marker='^')
    plt.errorbar(x=bin_centers, y=means97, linestyle='none', marker='s')
    plt.hlines([0.1, 0.3, 0.5, 0.7, 0.9, 0.97], bin_edges[0], bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/invquant_' + variable + '.pdf')


def plot_timeseries(df, suffix, title=''):
    plt.figure()
    df.index = df['date']
    df['y'].plot(style='r', label="sales")
    df['yhat_mean'].plot(style='b', label="prediction")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.png'.format(suffix))


def plotting(df, suffix=''):
    df = df[['y', 'yhat_mean', 'item', 'store', 'date']]

    ts_data = df.groupby(['date'])[['y', 'yhat_mean']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix, 'all')

    predictions_grouped = df.groupby(['store'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'store_' + str(name) + suffix)

    predictions_grouped = df.groupby(['item'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'item_' + str(name) + suffix)


def transform_nbinom(mean, var):
    p = np.minimum(np.where(var > 0, mean / var, 1 - 1e-8), 1 - 1e-8)
    n = np.where(var > 0, mean * p / (1 - p), 1)
    return n, p


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


def wasserstein_2(p, q, n_bins):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    wasser_distance = 2.0 * np.sum(np.abs(cdf_p - cdf_q)) / len(cdf_p)
    return wasser_distance * n_bins / (n_bins - 1.0)


def kullback_2(p, q):
    mask = p > 0  # limit x -> 0 of x log(x) is zero
    return np.float(np.sum(p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))))


def jensen_2(p, q):
    m = (p + q) / 2.0
    return (kullback_2(p, m) + kullback_2(q, m)) / 2.0


def kullback_e(p, q):
    mask = p > 0  # limit x -> 0 of x log(x) is zero
    return np.float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def jensen_e(p, q):
    m = (p + q) / 2.0
    return (kullback_e(p, m) + kullback_e(q, m)) / 2.0


def cdf_accuracy(cdf_truth, metric='wasserstein_2'):
    counts = cdf_truth.value_counts(bins=100)
    n_cdf_bins = len(counts)
    pmf = counts / np.sum(counts)  # relative frequencies for each bin
    unif = np.full_like(pmf, 1.0 / len(pmf))  # uniform distribution
    if metric == 'wasserstein_2':
        divergence = wasserstein_2(unif, pmf, n_cdf_bins)
    elif metric == 'kullback_2':
        divergence = kullback_2(unif, pmf)
    elif metric == 'kullback_e':
        divergence = kullback_e(unif, pmf)
    elif metric == 'jensen_2':
        divergence = jensen_2(unif, pmf)
    elif metric == 'jensen_e':
        divergence = jensen_e(unif, pmf)
    cdf_acc = np.float(1.0 - np.clip(divergence, a_min=None, a_max=1.0))
    print('cdf accuracy: {}'.format(cdf_acc))


def set_td_weights_cb(df):
    df['td'] = pd.to_timedelta(df['date']).dt.days
    df['td'] = df['td'] - df['td'].min()
    decay_const = 365. # number of days after the weight is down to 0.5
    const = -np.log(0.5) / decay_const
    df['weights_cb'] = np.exp((df['td'] - df['td'].max()) * const)
    return df


def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = set_td_weights_cb(df)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['weekofmonth'] = np.ceil(df['date'].dt.day / 7)
   
    y = np.asarray(df['sales'])
    X = df.drop(columns='sales')
    return X, y


def cb_mean_model():
    fp = {}
    fp['item'] = flags.IS_UNORDERED
    fp['store'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['month'] = flags.IS_ORDERED
    fp['weekofmonth'] = flags.IS_ORDERED
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR

    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=2),}

    features = [
        'store',
        ('store', 'td'),
        'item',
        ('item', 'store'),
        'dayofweek',
        'dayofyear',
        'month',
        ('weekofmonth', 'month'),
        ('weekofmonth', 'store'),
        ('dayofweek', 'store'),
        ('dayofweek', 'item'),
        ('store', 'month'),
        ('item', 'month'),
    ]

    plobs = [cyclic_boosting.observers.PlottingObserver(iteration=-1)]

    est = cyclic_boosting.CBFixedVarianceRegressor(
              feature_properties=fp,
              feature_groups=features,
              observers=plobs,
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
    fp = {}
    fp['item'] = flags.IS_UNORDERED
    fp['store'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['month'] = flags.IS_ORDERED
    fp['weekofmonth'] = flags.IS_ORDERED
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['yhat_mean_feature'] = flags.IS_CONTINUOUS

    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=2),}

    features = [
        'yhat_mean_feature',
        'store',
        ('store', 'td'),
        'item',
        ('item', 'store'),
        'dayofweek',
        'dayofyear',
        'month',
        ('weekofmonth', 'month'),
        ('weekofmonth', 'store'),
        ('dayofweek', 'store'),
        ('dayofweek', 'item'),
        ('store', 'month'),
        ('item', 'month'),
    ]

    plobs = [cyclic_boosting.observers.PlottingObserver(iteration=-1)]

    est = CBNBinomC(
        mean_prediction_column='yhat_mean',
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
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

    if horizon is None:
        truth_emov = X_grouped['y'].apply(lambda x: x.ewm(com=2.0, ignore_na=True).mean())
        pred_emov = X_grouped['yhat_mean'].apply(lambda x: x.ewm(com=2.0, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        X = X[X[date_col] == X[date_col].max()]
        return X[group_cols + ['correction_factor']]
    else:
        truth_emov = X_grouped['y'].apply(lambda x: x.shift(horizon).ewm(com=2.0, ignore_na=True).mean())
        pred_emov = X_grouped['yhat_mean'].apply(lambda x: x.shift(horizon).ewm(com=2.0, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        return X


def main(args):
    df = pd.read_csv('data/train.csv')

    X, y = prepare_data(df)

    split_date = '2017-07-01'

    ml_est_mean = cb_mean_model()
    ml_est_mean.fit(X[X['date'] < split_date].copy(), y[X['date'] < split_date])
    plot_CB('analysis_CB_mean_iterlast', [ml_est_mean.ests[-1][1].observers[-1]], ml_est_mean.ests[-2][1])

    X['yhat_mean'] = ml_est_mean.predict(X.copy())
#    plot_factors(ml_est_mean, X, [0,10000,50000])

    X['y'] = y
    horizon = 3
    if horizon is None:
        df_correction = emov_correction(X[X['date'] < split_date], ['store', 'item'], 'date')
        df_correction.reset_index(drop=True, inplace=True)
        X = X.merge(df_correction, on=['store', 'item'], how='left')
    else:
        X = emov_correction(X, ['store', 'item'], 'date', horizon)
    mask = (X['date'] >= split_date) & (pd.notna(X['correction_factor']))
    X['yhat_mean'][mask] = X['correction_factor'][mask] * X['yhat_mean'][mask]

    X['yhat_mean_feature'] = X['yhat_mean']

    ml_est_width = cb_width_model()
    mask = X['date'] >= split_date
    ml_est_width.fit(X[mask].copy(), np.asarray(X['y'])[mask])
#    plot_CB('analysis_CB_width_iterlast', [ml_est_width.ests[-1][1].observers[-1]], ml_est_width.ests[-2][1])

    c = ml_est_width.predict(X.copy())
    X['yhat_var'] = X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']
    X['n'], X['p'] = transform_nbinom(X['yhat_mean'], X['yhat_var'])
    plot_cdf(X['n'][50000], X['p'][50000])

    X['cdf_truth'] = nbinom.cdf(X['y'], X['n'], X['p'])
    cdf_accuracy(X['cdf_truth'], 'wasserstein_2')
    plot_cdf_truth(X['cdf_truth'][mask])
    plot_invquants(X[mask], 'yhat_mean')
    plot_invquants(X[mask], 'item')
    plot_invquants(X[mask], 'store')
    plot_invquants(X[mask], 'dayofweek')
    plot_invquants(X[mask], 'dayofyear')
    plot_invquants(X[mask], 'td')

    X['yhat_mean'] = np.round(X['yhat_mean'], 2)
    X['yhat_var'] = np.round(X['yhat_var'], 2)
    X[['item', 'store', 'date', 'y', 'yhat_mean', 'yhat_var']][mask].to_csv('forecasts_2017.csv', index=False)
    plotting(X[mask])
    eval_results(X['yhat_mean'][mask], X['y'][mask])

#    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
