import missing_value_pred as mvp
import pandas as pd

def cate2ind(df, key):
    """
    missing value will also be treated as a new value
    data: pandas dataframe
    key: the key of the specified column to convert
    """
    column = df[key]
    cate_map = mvp.isCategorical(column)
    print key
    print cate_map
    if (cate_map == None):
        return df

    if (len(cate_map)==2):  # if is binary, only leave one
        cate_map.pop(cate_map.keys()[0])
    # map -> ind with 0/1
    for category in cate_map:
        new_col = column.apply(lambda x: int(x==category)) # equal => 1, not equal => 0
        if (pd.isnull(category)):    # rename the nan
            category = "missing"
        new_name = key + "/" + str(category)
        df[new_name] = new_col

    df = df.drop(key, axis=1)    # delete the original column

    return df

def miss2ind(df, key):
    """
    convert a column from given dataframe to missing/not missing (1/0), drop the original one

    """
    column = df[key]
    new_col = pd.isnull(column)
    new_name = key + "/" + "missing"
    df[new_name] = new_col
    df = df.drop(key, axis=1)

    return df