import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
_logger = logging.getLogger(__name__)

def _discretize_by_width(col, num_bins, labels):
    maxvalue = col.max()
    minvalue = col.min()
    width = float((maxvalue-minvalue))/num_bins
    bins = [minvalue + x*width for x in range(num_bins)]+[maxvalue]
    if labels:
        if len(labels)!=num_bins:
            raise ValueError('Length of assigned labels not consistent with num_bins!')
        else:
            group_names = labels
    else:
        group_names = range(num_bins)
    return pd.cut(col, bins,labels=group_names, include_lowest=True)


def _discretize_by_frequency(col, num_bins, labels):
    percent = 1.0/num_bins
    bins = sorted(list(set(col.quantile([x*percent for x in range(num_bins+1)]))))
    if len(bins)-1 < num_bins:
        num_bins = len(bins)-1
        _logger.info('...Only %d bins (unbalanced) generated due to overlapping percentile boundaries.'%num_bins)
    if labels:
        if len(labels)!=num_bins:
            raise ValueError('Length of assigned labels not consistent with num_bins!')
        else:
            group_names = labels
    else:
        group_names = range(num_bins)
    return pd.cut(col, bins,labels=group_names, include_lowest=True)


def _discretize_by_kmeans(col, num_bins, random_state):
    nan_idx = col[col.isnull()].index
    kmeans = KMeans(n_clusters=num_bins, random_state=random_state)
    kmeans = kmeans.fit(col.dropna().values.T.reshape(-1, 1))
    group = kmeans.labels_
    if col.isnull().sum() > 0:
        group = group.astype(float)
        for idx in nan_idx:
            group = np.insert(group,idx,np.nan)
    return pd.Series(group)


def _discretize_by_gmm(col, num_bins, random_state):
    nan_idx = col[col.isnull()].index
    gmm = GaussianMixture(n_components=num_bins,covariance_type='full',random_state=random_state)
    gmm = gmm.fit(X=np.expand_dims(col.dropna(), 1))
    if col.isnull().sum() == 0:
        group = gmm.predict(X=np.expand_dims(col, 1))
    else:
        group = gmm.predict(X=np.expand_dims(col.dropna(), 1)).astype(float)
        for idx in nan_idx:
            group = np.insert(group,idx,np.nan)
    return pd.Series(group)


def discretize(col, num_bins=10, by='width', labels = None, random_state=0):
    if col.dropna().sum() == 0:
        raise ValueError('Empty column!')
    if by == 'width':
        return _discretize_by_width(col, num_bins, labels)

    elif by == 'frequency':
        return _discretize_by_frequency(col, num_bins, labels)

    elif by == 'kmeans':
        if labels:
            _logger.info('...Applying kmeans clustering, so user-defined labels are ignored.')
        return _discretize_by_kmeans(col, num_bins, random_state)

    elif by == 'gmm':
        if labels:
            _logger.info('...Applying gmm clustering, so user-defined labels are ignored.')
        return _discretize_by_gmm(col, num_bins, random_state)

    else:
        raise ValueError('...Invalid by (binning method) parameter %s'%by)
