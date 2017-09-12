import numpy as np
import pandas as pd

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
        

    def __init__(self, categorical_features='95in10', n_limit=10, text2int=True):
        
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
		print('Warning:',col.name,'is an empty column.')
		print('The encoder will discard it.')
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
        
        #if label is not None:
        #    self.label = label.columns[0]
        #    data_copy.drop(label.columns[0],axis=1,inplace=True)
        
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

        #if label is not None:
        #    set_columns = set(data_copy.drop(label,axis=1).columns)
        #else:
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
            encoded = pd.get_dummies(col, dummy_na=True, prefix=col.name)

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

    def fit_transform(self, data, label=None):
        self.fit(data, label)
        return self.transform(data, label)
