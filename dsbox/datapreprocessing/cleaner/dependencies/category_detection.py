import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def __shuffle_dict(col, seed):
    key = col.unique()
    shuffled = np.array(key, copy=True)
    shuffled = shuffle(shuffled, random_state=seed)
    adict = dict(zip(key, shuffled))
    return col.apply(lambda x: adict[x])


def __label(x):
    if x > 0.5:
        return 'H'
    elif x > 0.3:
        return 'M'
    return 'L'


def __safe_div(x):
    if x[1] == 0:
        return 0
    return x[0] / x[1]


def __tableGen(data):
    nlist = []
    table = pd.DataFrame(columns=[
        'col_name', 'nunique', 'nunique_ratio', 'H', 'M', 'L', 'ratio_H', 'ratio_M',
        'ratio_L', 'dropMean', 'dropMedian', 'dropMax', 'dropMin', 'dtype', '95%in10'])

    for name in data:
        col = data[name]
        if col.dtype.kind in np.typecodes['AllInteger'] + 'uf' and col.nunique() > 2:
            nlist.append(name)

    if len(nlist) > 1:
        ndata = data[nlist]  # New DataFrame
        origCorr = ndata.corr(method='spearman').fillna(0)  # New DataFrame
        ncol = len(nlist)
    else:
        nlist = []

    for name in data:
        col = data[name]
        # for empty column
        if col.count() == 0:
            table = table.append({'col_name': name, 'nunique': 0, 'nunique_ratio': 0,
                                  'H': 0, 'M': 0, 'L': 0,
                                  'ratio_H': 0, 'ratio_M': 0, 'ratio_L': 0,
                                  'dropMean': 0, 'dropMedian': 0, 'dropMax': 0, 'dropMin': 0,
                                  'dtype': -1, '95%in10': False}, ignore_index=True)

        # for numbers w/ nunique > 2 (when there are more then 1 such columns)
        elif name in nlist:
            in10 = col.value_counts().head(10).sum() / float(col.count()) > .95
            orig = abs(origCorr[name])
            temp = [1] * ncol
            level = list(orig.map(__label))
            lvl = (max(0, level.count('H') - 1), level.count('M'), level.count('L'))
            lvl_ratio = tuple(map(lambda x: round(float(x) / (len(level) - 1), 4), lvl))

            for i in range(5):
                ndatacopy = ndata.copy().fillna(0)  # New DataFrame
                ndatacopy[name] = __shuffle_dict(ndatacopy[name], i)
                corr = ndatacopy.corr(method='spearman')[name].fillna(0)  # New DataFrame
                shuf = abs(corr)
                temp = [min(x) for x in zip(shuf, temp)]

            p = orig - temp

            table = table.append({'col_name': name, 'nunique': col.nunique(),
                                  'nunique_ratio': round(float(col.nunique()), 4) / col.count(),
                                  'H': lvl[0], 'M': lvl[1], 'L': lvl[2],
                                  'ratio_H': lvl_ratio[0], 'ratio_M': lvl_ratio[1], 'ratio_L': lvl_ratio[2],
                                  'dropMean': round(np.mean(p), 4), 'dropMedian': round(np.median(p), 4),
                                  'dropMax': round(np.max(p), 4), 'dropMin': round(np.min(p), 4),
                                  'dtype': col.dtype, '95%in10': in10}, ignore_index=True)

        # for objects (and numbers when there are few numerical column)
        else:
            in10 = col.value_counts().head(10).sum() / float(col.count()) > .95
            table = table.append({'col_name': name, 'nunique': col.nunique(),
                                  'nunique_ratio': round(float(col.nunique()), 4) / col.count(),
                                  'H': 0, 'M': 0, 'L': 0,
                                  'ratio_H': 0, 'ratio_M': 0, 'ratio_L': 0,
                                  'dropMean': 0, 'dropMedian': 0,
                                  'dropMax': 0, 'dropMin': 0,
                                  'dtype': col.dtype, '95%in10': in10}, ignore_index=True)
    return table


def __column_detect(dtype, nunique, nunique_ratio, dropMax, H, M):
    if dtype in [float, np.float64, np.float32, np.float16]:
        return False
    elif dtype in [int, np.int64, np.int32, np.int16, np.int8]:
        if nunique < 50 and nunique_ratio < 0.7:
            return True
        else:
            return False
    else:
        if nunique > 16:
            return False
        else:
            if dropMax <= 0.05:
                return True
            else:
                if H + M < 1:
                    return True
                else:
                    if nunique <= 10:
                        return True
                    return False


def category_detect(data):
    table = __tableGen(data)
    res_dict = {}
    for index, row in table.iterrows():
        res = __column_detect(row['dtype'], row['nunique'], row['nunique_ratio'], row['dropMax'], row['H'], row['M'])
        res_dict[row['col_name']] = res
    return res_dict
