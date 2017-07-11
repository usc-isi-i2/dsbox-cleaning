import numpy as np
import pandas as pd
from dsbox.datapreprocessing.profiler import profile_data

def isCategorical(col):
    """ 
    hardcoded rule for identifying (integer) categorical column
    """
    return col.value_counts().head(10).sum() / float(col.count()) > .95


def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])


def onehot_encode(col, exist_nan, limited=True):
    """
    convert specified column into multiple columns with 0/1 indicators
    if limited=True, only convert top t values into columns and the rest as one column
    t: most frequent t values that sum up to 95% of data
    """
    if limited:
        cdf = (col.value_counts() / float(col.count())).cumsum()
        _col = col.replace(cdf[cdf>.95].index, 'others')
        return pd.get_dummies(_col, prefix=col.name, dummy_na=exist_nan)
    else:    
        return pd.get_dummies(col,prefix=col.name, dummy_na=exist_nan)

def encode(data_path):
    """

    """
    del_col = []
    new_col = []
    isDF = True

    ## csv as input ##
    if not isinstance(data_path, pd.DataFrame):
        isDF = False
        data = pd.read_csv(data_path)
        data_raw = pd.read_csv(data_path, dtype=str)
    ## data frame as input ##
    else:
        data = data_path

    for column_name in data:
        col = data[column_name].copy()
        
        exist_nan = (col.isnull().sum() > 0)

        # empty column (all missing/NaN)
        if col.count() == 0:
            print '...Delete *'+str(column_name)+"* column: empty column."
            del_col.append(col.name)

        # dtype = integer
        elif col.dtype.kind in np.typecodes['AllInteger']+'u':
            if isCategorical(col):
                print "...Delete *"+str(column_name)+"* column: integer category." 
                del_col.append(col.name)
                print "...Insert columns to onehot encode *"+str(column_name)+"*."
                new_col.append(onehot_encode(col, exist_nan))

        # dtype = float from csv file 
        # (check if it's int column with missing value)
        elif col.dtype.kind == 'f':
            if not isDF:
                col = data_raw[column_name].copy()
                pf = profile_data(pd.DataFrame(col))[column_name]['numeric_stats']
                if ('integer' in pf and 'decimal' not in pf):
                    if isCategorical(col):
                        print '...Delete *'+str(column_name)+'* column: integer category.'
                        del_col.append(col.name)
                        print "...Insert columns to onehot encode *"+str(column_name)+"*."
                        new_col.append(onehot_encode(col,exist_nan))
        
        # dtype = category
        elif col.dtype.name == 'category':
            print '...Delete *'+column_name+'* column: category dtype.'
            del_col.append(col.name)
            print "...Insert columns to onehot encode *"+str(column_name)+"*."
            new_col.append(onehot_encode(col,exist_nan))

        # for other dtypes
        else:
            #col = col.astype(str)
            if isCategorical(col):
                print '...Delete *'+str(column_name)+'* column: object/other category.'
                del_col.append(col.name)
                print "...Insert columns to onehot encode *"+str(column_name)+"*."
                new_col.append(onehot_encode(col,exist_nan))
            else:
                print '...Convert *'+str(column_name)+'* column from text to integer codes.'
                del_col.append(col.name)
                new_col.append(text2int(col))

    # drop empty, category-like and those converted to int codes
    rest = data.drop(del_col, axis=1)

    # insert columns from onehot encoding and text2int codes
    result = pd.concat(new_col+[rest], axis=1)  

    return result
