import numpy as np
import pandas as pd
from dsbox.datapreprocessing.profiler import profile_data

def isCat_95in10(col):
    """
    hardcoded rule for identifying (integer/string) categorical column
    """
    return col.value_counts().head(10).sum() / float(col.count()) > .95


def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])


## not used - would remove later
def onehot_encode(col, exist_nan, limited=True):
    """
    convert specified column into multiple columns with 0/1 indicators
    if limited=True, and there are more than 10 distinct values,
    only convert top t values into columns and the rest as one column
    t: most frequent t values that sum up to 95% of data
    """
    if col.nunique() == 1:
        if exist_nan:
            new_col = col.isnull().astype(int)
            new_col.name = col.name+"_nan"
            return pd.DataFrame(new_col)
        else:
            new_col = col.astype('category').cat.codes
            return pd.DataFrame(new_col, columns=[col.name])
    elif col.nunique() == 2 and not exist_nan:
        return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])

    elif limited and col.nunique() > 10:
        #cdf = (col.value_counts() / float(col.count())).cumsum()
        #_col = col.replace(cdf[cdf>.95].index, 'others')
        top10 = col.value_counts().head(10).index
        _col = col.apply(lambda x: x if x in top10 else 'others')
        return pd.get_dummies(_col, prefix=col.name, dummy_na=exist_nan)
    else:
        return pd.get_dummies(col,prefix=col.name, dummy_na=exist_nan)


## not used - would remove later
def encode(data_path,label=None):
    """
    take pandas dataframe or raw csv file as input.
    return a dataframe with one-hot encoded indicator columns.
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

    # if label is in the dataset, specify label=label_name
    # process seperately
    if label:
        label_col = data[label]
        data = data.drop(label,axis=1)

    for column_name in data:
        col = data[column_name].copy()

        exist_nan = (col.isnull().sum() > 0)

        # empty column (all missing/NaN)
        if col.count() == 0:
            print('...Delete *'+str(column_name)+"* column: empty column.")
            del_col.append(col.name)

        # dtype = integer
        elif col.dtype.kind in np.typecodes['AllInteger']+'u':
            if isCategorical(col):
                print("...Delete *"+str(column_name)+"* column: integer category.")
                del_col.append(col.name)
                print("...Insert columns to onehot encode *"+str(column_name)+"*.")
                new_col.append(onehot_encode(col, exist_nan))

        # dtype = float from csv file
        # (check if it's int column with missing value)
        elif col.dtype.kind == 'f':
            if not isDF:
                col = data_raw[column_name].copy()
                pf = profile_data(pd.DataFrame(col))[column_name]['numeric_stats']
                if ('integer' in pf and 'decimal' not in pf):
                    if isCategorical(col):
                        print('...Delete *'+str(column_name)+'* column: integer category.')
                        del_col.append(col.name)
                        print("...Insert columns to onehot encode *"+str(column_name)+"*.")
                        new_col.append(onehot_encode(col,exist_nan))

        # dtype = category
        elif col.dtype.name == 'category':
            print('...Delete *'+column_name+'* column: category dtype.')
            del_col.append(col.name)
            print("...Insert columns to onehot encode *"+str(column_name)+"*.")
            new_col.append(onehot_encode(col,exist_nan))

        # for other dtypes
        else:
            #col = col.astype(str)
            if isCategorical(col):
                print('...Delete *'+str(column_name)+'* column: object/other category.')
                del_col.append(col.name)
                print("...Insert columns to onehot encode *"+str(column_name)+"*.")
                new_col.append(onehot_encode(col,exist_nan))
            else:
                print('...Convert *'+str(column_name)+'* column from text to integer codes.')
                del_col.append(col.name)
                new_col.append(text2int(col))

    # for label column
    if label:
        if label_col.dtype.kind not in 'biufcmM':
            nc = pd.Series(label_col.astype('category').cat.codes,name=label_col.name)
            new_col.append(nc)
        else:
            new_col.append(label_col)

    # drop empty, category-like and those converted to int codes
    rest = data.drop(del_col, axis=1)

    # insert columns from onehot encoding and text2int codes
    result = pd.concat([rest]+new_col, axis=1)

    return result


class Encoder(object):
    """
    An one-hot encoder, which
    
    1. is given rules or values to identify categorical columns/
       - categorical_features:
            '95in10': is category if 95% of the column fall into 10 values.
       - tex2int: if non-categorical text/string columns be mapped to integers
       - n_limit: max number of distinct values to one-hot encode,
         remaining values with fewer occurence are put in [colname]_other_ column.
    
    2. uses data given in fit() function to tune the encoder.
    
    3. transform(): input data to be encoded and output the result.
    """
    
    def __repr__(self):
        return "%s(%r)" % ('Encoder', self.__dict__)
        

    def __init__(self, categorical_features='95in10', n_limit=10, text2int=False):
        
        self.label = None
        self.categorical_features = categorical_features
        self.text2int = text2int
        self.table = None
        self.n_limit = n_limit
        self.columns = None
        self.empty = []
        
    
    def __column_features(self, col, n_limit):
        
        topn = col.dropna().unique()
        if n_limit:
            if col.nunique() > n_limit:
	        topn = col.value_counts().head(n_limit).index
        return col.name, list(topn)+['other_']
    
                
    def __process(self, col, categorical_features, n_limit):
	
        if categorical_features == '95in10':		
	    
            # if empty column (all missing/NaN)
	    if col.count() == 0:
		print 'Warning:',col.name,'is an empty column.'
		print 'The encoder will discard it.'
		self.empty.append(col.name)
                return
				
	    # if dtype = integer
	    elif col.dtype.kind in np.typecodes['AllInteger']+'u':
		if isCat_95in10(col):
		    return self.__column_features(col, n_limit)
	    
            # if dtype = category	
	    elif col.dtype.name == 'category':
		return self.__column_features(col, n_limit)
            
            # for the rest other than float
	    elif col.dtype.name not in np.typecodes['AllFloat']:
		if isCat_95in10(col):
		    return self.__column_features(col, n_limit)
	    
            return 
            

    def fit(self, data, label=None):
        """
        Feed in data set to fit, e.g. trainData. 
        The encoder would record categorical columns identified and 
        the corresponding (with top n occurrence) column values to 
        one-hot encode later in the transform step.
        """
        ## csv as input, otherwise data frame as input ##
        if not isinstance(data, pd.DataFrame):
            data = pd.read_csv(data)
        
        data_copy = data.copy()
        
        if label:
            self.label = label
            data_copy.drop(label,axis=1,inplace=True)
        
        self.columns = set(data_copy.columns)

        if self.categorical_features == '95in10':
            idict = {}
            for column_name in data_copy:
                col = data_copy[column_name]
                p = self.__process(col, self.categorical_features, self.n_limit)
                if p:
                    idict[p[0]] = p[1]
            self.table = idict
        return self

    def transform(self, data, label=None):
        """
        Convert and output the input data into encoded format,
        using the trained (fitted) encoder.
        Notice that a [colname]_other_ column is always kept for 
        one-hot encoded columns.
        Missing(NaN) cells in a column one-hot encoded would give 
        out a row of all-ZERO columns for the target column.
        """
        ## csv as input, otherwise data frame as input ##
        if not isinstance(data, pd.DataFrame):
            data = pd.read_csv(data)
        
        data_copy = data.copy()
        data_enc = data_copy[self.table.keys()]
        data_else = data_copy.drop(self.table.keys(),axis=1)

        if label:
            set_columns = set(data_copy.drop(label,axis=1).columns)
        else:
            set_columns = set(data_copy.columns)

        if set_columns != self.columns:
            raise ValueError('Columns(features) fed at transform() differ from fitted data.')
        
        data_enc = data_copy[self.table.keys()]
        data_else = data_copy.drop(self.table.keys(),axis=1)

        res = []
        for column_name in data_enc:
            col = data_enc[column_name]
            chg_v = lambda x: 'other_' if (x and x not in self.table[col.name]) else x
            col = col.apply(chg_v)
            encoded = pd.get_dummies(col, dummy_na=False, prefix=col.name)

            missed = (["%s_%s"%(col.name,str(i)) for i in self.table[col.name] if 
                    "%s_%s"%(col.name,str(i)) not in list(encoded.columns)])
            
            for m in missed:
                encoded[m] = 0
            
            res.append(encoded)
        
        data_else.drop(self.empty, axis=1, inplace=True)
        if self.text2int:
            for column_name in data_else:
                if data_else[column_name].dtype.kind not in np.typecodes['AllInteger']+'uf':
                    data_else[column_name] = text2int(data_else[column_name])
        
        res.append(data_else)
        result = pd.concat(res, axis=1)
        return result


