"""
Cond-dist predictions for the California Housing dataset.

Source and problem description:

* http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
* Hastie, The Elements of Statistical Learning, 10.14.1


Why are feature bins filled with **unnecessarily heterogeneous data** harmful
in prediction projects?

* Many machine-learning algorithms such as Neurobayes without internal boost
  can only consider axis-parallel cuts through the feature space.

  Often target averages inside these subspaces are used for the prediction,
  in particular averages over such heterogeneous feature bins.

* Such averages may not be meaningful or even **misleading**. For example, if
  you lumped together apples and peas, then the average size is neither a good
  predictor for apples nor for peas.

  Taking the median is not much better.

* Heterogeneous feature bins often lead to the famous **XOR problem** in
  machine learning which is notoriously hard to solve for many
  machine learning algorithms.

* In **power-law distributions**, taking averages is very misleading, e.g. if
  you lump together Harry Potter books with specialist books about twelve tone
  music, then taking the average sales leads to very bad predictions.
"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import cPickle

import matplotlib.pylab as plt
import matplotlib.ticker as ticker

import sklearn.datasets
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.ensemble

from nbpy import cyclic_boosting, pipeline, x_transformer, binning
from nbpy.cyclic_boosting.plots import plot_analysis
from nbpy import flags, meta_estimator, on_cols
from nbpy.smoothing.onedim import IsotonicRegressor

from nbpy.ext import predict_as_transform, plots


USE_PICKLE = False
GRADBOOST = False


def get_data(test_size):
    """Get the data for training and test.

    For the parameters, see :func:`sklearn.cross_validation.train_test_split`.

    :return: ``X_train, X_test, y_train, y_test``, the X's being Pandas
        DataFrames.
    """
    cal_housing = sklearn.datasets.fetch_california_housing()

    X_train, X_test, y_train, y_test = (
        sklearn.cross_validation.train_test_split(
            cal_housing.data, cal_housing.target, random_state=1))

    X_train = pd.DataFrame(X_train, columns=cal_housing.feature_names)
    X_test = pd.DataFrame(X_test, columns=cal_housing.feature_names)

    print("########## Describe calls:")
    print("########## X_train.describe():\n", X_train.describe())
    print("########## y_train.describe():\n", pd.Series(y_train).describe())

    print("##########11", np.isnan(np.asarray(X_train)).sum(axis=0))
    print(X_train[:10])
    return X_train, X_test, y_train, y_test


def add_quotients_of_features(X):
    name = 'AveOccupPerRoom'
    X[name] = X['AveOccup'] / X['AveRooms']
    assert np.all(np.unique(X['AveRooms']) > 0)
    name = 'RatioBedrms'
    X[name] = X['AveBedrms'] / X['AveRooms']
    assert np.all(np.unique(X['AveRooms']) > 0)


def prepare_features(X, fit_mode):
    """Prepare the features for the California Housing project and set the
    feature properties for Neurobayes.
    """
    print("##########7 features before preparation:", X.columns)

    add_quotients_of_features(X)

    del X['Latitude']
    del X['Longitude']

    print("##########7 features after preparation:", X.columns)

    return X


def create_pipeline():
    feature_properties = {}

    nearest_neighbors = sklearn.neighbors.KNeighborsRegressor(
        n_neighbors=25)

    nearest_neighbors = predict_as_transform.PredictAsTransform(
        ('avg_of_nearest_neighbors', nearest_neighbors))

    nearest_neighbors = on_cols.OnCols(
        ('Latitude', 'Longitude'), nearest_neighbors)

    preparer = x_transformer.CustomXTransformer(prepare_features)

    columns = ["MedInc",
               "AveOccup",
               "AveRooms",
               "AveBedrms",
               "Population",
               "HouseAge",
               "AveOccupPerRoom",
               "RatioBedrms",
               "avg_of_nearest_neighbors"]
    for column in columns:
        feature_properties[column] = flags.IS_CONTINUOUS

    features = feature_properties.keys()

    # add all 2D combinations
    for i, j in enumerate(columns):
        features += [(j, k) for k in columns[i+1:]]

    plobsfirst = cyclic_boosting.observers.PlottingObserver(iteration=1)
    plobslast = cyclic_boosting.observers.PlottingObserver(iteration=-1)
    plobs = [plobsfirst, plobslast]

    binner_few = binning.BinNumberTransformer(n_bins=30, feature_properties={'AveOccup': 1, 'AveBedrms': 1, 'AveOccupPerRoom': 1, 'AveRooms': 1, 'Population': 1, 'RatioBedrms': 1})
    binner_many = binning.BinNumberTransformer(n_bins=52, feature_properties={'MedInc': 1, 'avg_of_nearest_neighbors': 1, 'HouseAge': 1})

    explicit_smoothers = {
        ('avg_of_nearest_neighbors',): IsotonicRegressor(increasing=True),
        ('MedInc',): IsotonicRegressor(increasing=True),
        ('AveOccup',): IsotonicRegressor(increasing=False)
    }

    regressor = cyclic_boosting.regression.CBFixedVarianceRegressor(
        feature_properties=feature_properties,
        feature_groups=features,
        learn_rate=cyclic_boosting.learning_rate.logistic_learn_rate,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=cyclic_boosting.common_smoothers.SmootherChoiceSVD(explicit_smoothers=explicit_smoothers),
        aggregate=True
    )

    return pipeline.Pipeline([
        ('nearest_neighbors', nearest_neighbors),
        ('prepare_features', preparer),
        ('binner', binner_few),
        ('binner_many', binner_many),
        ('cyclic_boosting', regressor),
    ])


def create_pipeline_gb():
    nearest_neighbors = sklearn.neighbors.KNeighborsRegressor(
        n_neighbors=25)

    nearest_neighbors = predict_as_transform.PredictAsTransform(
        ('avg_of_nearest_neighbors', nearest_neighbors))

    nearest_neighbors = on_cols.OnCols(
        ('Latitude', 'Longitude'), nearest_neighbors)

    preparer = x_transformer.CustomXTransformer(prepare_features)

    regressor = sklearn.ensemble.GradientBoostingRegressor(
        alpha=0.9, init=None, learning_rate=0.1, loss='ls', max_depth=6, max_features=1.0, max_leaf_nodes=None,
        min_samples_leaf=3, min_samples_split=2, n_estimators=100, random_state=None, subsample=1.0, verbose=0,
        warm_start=False)

    return pipeline.Pipeline([
        ('nearest_neighbors', nearest_neighbors),
        ('prepare_features', preparer),
        ('cyclic_boosting', regressor),
    ])


def training(X_train, y_train):
    """Training of the estimator for the California Housing problem.
    """
    if not GRADBOOST:
        est = create_pipeline()
    else:
        est = create_pipeline_gb()

    boolean_index = np.asarray((X_train["HouseAge"] < 50) &
                               (X_train["AveOccup"] < 5))
    interaction_plot(X_train["HouseAge"][boolean_index],
                     X_train["AveOccup"][boolean_index],
                     y_train[boolean_index],
                     binsizeX=1.,
                     binsizeY=0.1)
    interaction_plot(X_train["Longitude"],
                     X_train["Latitude"],
                     y_train,
                     binsizeX=0.05,
                     binsizeY=0.05)
    profile_plot_saved(X_train["Longitude"], y_train)
    profile_plot_saved(X_train["Latitude"], y_train)

    est.fit(X_train, y_train)

    return est


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj=filename + "_{}".format(i), use_tightlayout=False,
            binners=binner,
            plot_yp = True
        )


def scoring(y_test, Ypred):
    def mad():
        return np.abs(y_test - Ypred).mean()
    print("########## Mean absolute deviation between prediction and y:", mad())

    def mse():
        return ((y_test - Ypred) ** 2).mean()
    print("########## Mean squared error between prediction and y:", mse())

    def rsquared():
        return 1 - (((y_test - Ypred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    print("########## R squared:", rsquared())


def griddata(x, y, z, binsizeX, binsizeY):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi = np.arange(xmin, xmax + binsizeX, binsizeX)
    yi = np.arange(ymin, ymax + binsizeY, binsizeY)
    xi, yi = np.meshgrid(xi, yi)

    grid = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape

    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]
            yc = yi[row, col]

            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsizeX / 2., posy < binsizeY / 2.)
            ind = np.where(ibin == True)[0]

            b = z[ind]
            if b.size != 0:
                binval = np.median(b)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan

    return xi, yi, grid


def interaction_plot(x, y, z, binsizeX, binsizeY):
    """ Draw mean ``y`` as a function of x1, x2
    """
    xi, yi, zi = griddata(x, y, z, binsizeX=binsizeX, binsizeY=binsizeY)

    plt.contour(xi, yi, zi, 5, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 5)
    plt.colorbar()  # draw colorbar
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.savefig('y_{0}_{1}.png'.format(x.name, y.name))
    plt.clf()


def profile_plot_saved(x, y, ):
    """ Draw mean ``y`` as a function of x
    """
    plots.profile_plot(x, y)
    plt.xlabel(x.name)
    plt.savefig('y_{0}.png'.format(x.name))
    plt.clf()


def plot_target(y, suffix):
    plt.hist(y, bins=100)
    plt.savefig('y_' + suffix + '.png')
    plt.clf()


def plot_factors(est, X_test):
    factors = {}
    for feature in est.ests[-1][1].features:
        ft = "" if feature.feature_type is None else "_{}".format(
            feature.feature_type)

        factors[
            "_".join(feature.feature_group) + ft
            ] = np.exp(est.ests[-1][1]._pred_feature(X_test, feature, is_fit=False)) - 1.
    df_factors = pd.DataFrame(factors)

    ax = df_factors.loc[[0,1,2]].plot(kind='barh', title="factors", legend=True, fontsize=15)
    fig = ax.get_figure()
    ax.axvline(0, color="gray")
    ax.set_yticks([])

    def setfactorformat(x, pos):
        return '%1.1f' % (x + 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(setfactorformat))

    fig.savefig('factors.pdf')


def main():
    """Main program"""
    X_train, X_test, y_train, y_test = get_data(0.3)
    X_train_pred = X_train.copy()
    est = training(X_train, y_train)

    if not GRADBOOST:
        plotfilename = 'analysis_CB_grossmargin_iterlast'
        plot_CB(plotfilename, [est.ests[-1][1].observers[-1]], [est.ests[-2][1], est.ests[-2][1]])

    pickle_file = 'california_housing.est'

    if USE_PICKLE:
        with open(pickle_file, 'w') as fout:
            cPickle.dump(est, fout)

        with open(pickle_file) as fin:
            est = cPickle.load(fin)

    Ypred_train = est.predict(X_train_pred)
    Ypred = est.predict(X_test)

    if not GRADBOOST:
        plot_factors(est, X_test)

    # cut off at 5.
    cut_off = 5.
    Ypred[Ypred > cut_off] = cut_off
    Ypred_train[Ypred_train > cut_off] = cut_off

    print("scoring on test data:")
    scoring(y_test, Ypred)
    print("scoring on train data:")
    scoring(y_train, Ypred_train)

    plot_target(y_train, 'train')
    plot_target(y_test, 'test')
    plot_target(Ypred, 'pred_test')
    plot_target(Ypred_train, 'pred_train')


if __name__ == '__main__':
    main()
