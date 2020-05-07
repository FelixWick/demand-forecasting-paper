import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from nbpy import binning, pipeline, cyclic_boosting
from nbpy import flags
from nbpy.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from nbpy.cyclic_boosting.plots import plot_analysis

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj="plots/" + filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


def plot_factors(est, X):
    factors = {}
    for feature in est.ests[-1][1].features:
        ft = "" if feature.feature_type is None else "_{}".format(
            feature.feature_type)

        factors[
            "_".join(feature.feature_group) + ft
            ] = np.exp(est.ests[-1][1]._pred_feature(X, feature, is_fit=False)) - 1.
    df_factors = pd.DataFrame(factors)

    ax = df_factors.loc[[0,10000,50000]].plot(kind='barh', title="factors", legend=True, fontsize=15)
    fig = ax.get_figure()
    ax.axvline(0, color="gray")
    ax.set_yticks([])

    def setfactorformat(x, pos):
        return '%1.2f' % (x + 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(setfactorformat))

    fig.savefig('plots/factors.pdf')


def plot_timeseries(df, suffix, title=''):
    plt.figure()
    df.index = df['date']
    df['y'].plot(style='r', label="sales")
    df['yhat'].plot(style='b', label="prediction")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.png'.format(suffix))


def plotting(df, suffix=''):
    df = df[['y', 'yhat', 'item', 'store', 'date']]

    ts_data = df.groupby(['date'])[['y', 'yhat']].sum().reset_index()
    plot_timeseries(ts_data, 'full' + suffix, 'all')

    predictions_grouped = df.groupby(['store'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat'].sum().reset_index()
        plot_timeseries(ts_data, 'store_' + str(name) + suffix)

    predictions_grouped = df.groupby(['item'])
    for name, group in predictions_grouped:
        ts_data = group.groupby(['date'])['y', 'yhat'].sum().reset_index()
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


def cb_model():
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
#              weight_column="weights_cb",
              maximal_iterations=50,
              smoother_choice=cyclic_boosting.common_smoothers.SmootherChoiceGroupBy(
                  use_regression_type=True,
                  use_normalization=False,
                  explicit_smoothers=explicit_smoothers),
#              learn_rate=cyclic_boosting.learning_rate.logistic_learn_rate
          )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = pipeline.Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def emov_correction(X, group_cols, date_col, horizon=None):
    X.sort_values([date_col], inplace=True)
    X_grouped = X.groupby(group_cols)

    if horizon is None:
        truth_emov = X_grouped['y'].apply(lambda x: x.ewm(com=2.0, ignore_na=True).mean())
        pred_emov = X_grouped['yhat'].apply(lambda x: x.ewm(com=2.0, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        X = X[X[date_col] == X[date_col].max()]
        return X[group_cols + ['correction_factor']]
    else:
        truth_emov = X_grouped['y'].apply(lambda x: x.shift(horizon).ewm(com=2.0, ignore_na=True).mean())
        pred_emov = X_grouped['yhat'].apply(lambda x: x.shift(horizon).ewm(com=2.0, ignore_na=True).mean())
        X['correction_factor'] = truth_emov / pred_emov
        return X


def main(args):
    df = pd.read_csv('data/train.csv')

    X, y = prepare_data(df)

    ml_est = cb_model()

    split_date = '2017-01-01'
    final_date = '2017-03-31'

    ml_est.fit(X[X['date'] < split_date].copy(), y[X['date'] < split_date])

    plot_CB('analysis_CB_iterlast', [ml_est.ests[-1][1].observers[-1]], ml_est.ests[-2][1])

    yhat = ml_est.predict(X.copy())

    plot_factors(ml_est, X)

    df = X
    df['y'] = y
    df['yhat'] = np.round(yhat, 2)

#     horizon = None
#     if horizon is None:
#         df_correction = emov_correction(df[df['date'] < split_date], ['store', 'item'], 'date')
#         df_correction.reset_index(drop=True, inplace=True)
#         df = df.merge(df_correction, on=['store', 'item'], how='left')
#     else:
#         df = emov_correction(df, ['store', 'item'], 'date', horizon)
#     mask = (df['date'] >= split_date) & (pd.notna(df['correction_factor']))
#     df['yhat'][mask] = df['correction_factor'][mask] * df['yhat'][mask]

    df[['item', 'store', 'date', 'y', 'yhat']][df['date'] >= split_date].to_csv('forecasts_2017.csv', index=False)

    plotting(df[df['date'] >= split_date])

    mask = (df['date'] >= split_date) & (df['date'] <= final_date)
    eval_results(df['yhat'][mask], df['y'][mask])

#    embed()


if __name__ == "__main__":
    main(sys.argv[1:])

