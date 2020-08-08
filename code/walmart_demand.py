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


def plot_pdf(n, p):
    plt.figure()
    x = range(30)
    plt.plot(x, nbinom.pmf(x, n, p))
    plt.tight_layout()
    plt.savefig('plots/pdf.pdf')


def plot_cdf(n, p):
    plt.figure()
    x = np.linspace(0.0, 30.0, num=100)
    plt.plot(x, nbinom.cdf(x, n, p))
    plt.tight_layout()
    plt.savefig('plots/cdf.pdf')


def plot_cdf_truth(cdf_truth, suffix):
    plt.figure()
    plt.hist(cdf_truth, bins=30)
    plt.tight_layout()
    plt.savefig('plots/cdf_truth_' + suffix + '.pdf')


def plot_invquants(X, variable, suffix):
    means_result = binned_statistic(X[variable], [X['cdf_truth']<=0.1, X['cdf_truth']<=0.3, X['cdf_truth']<=0.5, X['cdf_truth']<=0.7, X['cdf_truth']<=0.9, X['cdf_truth']<=0.97], bins=20, statistic='mean')
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
    plt.savefig('plots/invquant_' + variable + '_' + suffix + '.pdf')


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
    df = df[['y', 'yhat_mean', 'item_id', 'store_id', 'date']]

    ts_data = df.groupby(['date'])[['y', 'yhat_mean']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix, 'all')

    predictions_grouped = df.groupby(['store_id'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'store_' + str(name) + suffix)

    predictions_grouped = df.groupby(['item_id'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat_mean'].sum().reset_index()
        plot_timeseries(ts_data, 'item_' + str(name) + suffix)


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
            for event_days in range(-3, 1):
                df.loc[df['date'] == pd.to_datetime(event_date).date() + datetime.timedelta(days=event_days), event] = event_days

    return df


def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = set_td_weights_cb(df)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['weekofmonth'] = np.ceil(df['date'].dt.day / 7)

    df['snap'] = 0
    df['snap'] = df['snap_CA'][df['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]
    df['snap'] = df['snap_TX'][df['store_id'].isin(['TX_1', 'TX_2', 'TX_3'])]
    df['snap'] = df['snap_WI'][df['store_id'].isin(['WI_1', 'WI_2', 'WI_3'])]

    df['price_ratio'] = df['sell_price'] / df['list_price']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)

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
    return fp


def cb_mean_model():
    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                         }

    features = [
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
                         }

    features = [
#        'yhat_mean_feature',
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
    return ml_est_mean


def mean_predict(X, ml_est_mean):
    yhat_mean = ml_est_mean.predict(X.copy())
#    plot_factors(ml_est_mean, X, [359323])
    return yhat_mean


def width_fit(X, split_date):
    ml_est_width = cb_width_model()
    mask = X['date'] >= split_date
    ml_est_width.fit(X[mask].copy(), np.asarray(X['y'])[mask])
    plot_CB('analysis_CB_width_iterlast', [ml_est_width.ests[-1][1].observers[-1]], ml_est_width.ests[-2][1])
    return ml_est_width


def width_predict(X, ml_est_width):
    c = ml_est_width.predict(X.copy())
#    c[X['yhat_mean'] < 1.] = 1
    return X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']


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
    plot_pdf(X['n'][359323], X['p'][359323])
    plot_cdf(X['n'][359323], X['p'][359323])
    X['cdf_truth'] = random_from_cdf_interval(X, mode='nbinom')
    X['cdf_truth_poisson'] = random_from_cdf_interval(X, mode='poisson')
    return X


def main(args):
    df = pd.read_csv('data/M5_FOODS_3_5_2013.csv')

    X, y = prepare_data(df)

    split_date = '2016-01-01'

    X_mean_fit = X.copy()
    ml_est_mean = mean_fit(X_mean_fit, y, split_date)
    del X_mean_fit

    X_mean_pred = X.copy()
    X['yhat_mean'] = mean_predict(X_mean_pred, ml_est_mean)
    del X_mean_pred

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

#    X['yhat_mean_feature'] = X['yhat_mean']

    X_width_fit = X.copy()
    ml_est_width = width_fit(X_width_fit, split_date)
    del X_width_fit

    X_width_predict = X.copy()
    X['yhat_var'] = width_predict(X_width_predict, ml_est_width)
    del X_width_predict

    X = cdf_truth(X)

    mask = X['date'] >= split_date
    cdf_accuracy(X['cdf_truth'][mask], 'wasserstein_2')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'wasserstein_2')
    cdf_accuracy(X['cdf_truth'][mask], 'kullback_2')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'kullback_2')
    cdf_accuracy(X['cdf_truth'][mask], 'kullback_e')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'kullback_e')
    cdf_accuracy(X['cdf_truth'][mask], 'jensen_2')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'jensen_2')
    cdf_accuracy(X['cdf_truth'][mask], 'jensen_e')
    cdf_accuracy(X['cdf_truth_poisson'][mask], 'jensen_e')

    plot_cdf_truth(X['cdf_truth'][mask], 'nbinom')
    plot_cdf_truth(X['cdf_truth_poisson'][mask], 'poisson')
    plot_cdf_truth(X['cdf_truth'][mask & (X['yhat_mean'] >= 1.)], 'nbinom_larger1')
    plot_cdf_truth(X['cdf_truth_poisson'][mask & (X['yhat_mean'] >= 1.)], 'poisson_larger1')

    plot_invquants(X[mask], 'yhat_mean', 'nbinom')
    plot_invquants(X[mask], 'store_id', 'nbinom')
    plot_invquants(X[mask], 'dayofweek', 'nbinom')
    plot_invquants(X[mask], 'yhat_mean', 'poisson')
    plot_invquants(X[mask], 'store_id', 'poisson')
    plot_invquants(X[mask], 'dayofweek', 'poisson')

    X['yhat_mean'] = np.round(X['yhat_mean'], 2)
    X['yhat_var'] = np.round(X['yhat_var'], 2)
    X[['item_id', 'store_id', 'date', 'y', 'yhat_mean', 'yhat_var']][mask].to_csv('forecasts_2016.csv', index=False)

    plotting(X[mask])
#    plotting(X)

    eval_results(X['yhat_mean'][mask], X['y'][mask])

#    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
