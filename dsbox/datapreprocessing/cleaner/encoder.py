import numpy as np
import pandas as pd
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import NamedTuple, Sequence

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


Input = pd.DataFrame
Output = pd.DataFrame

Params = NamedTuple('Params', [
    ('mapping', dict),
    ('all_columns', list),
    ('empty_columns', list)
    ])


class Encoder(UnsupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    An one-hot encoder, which

    1. is given rules or values to identify categorical columns/
       - categorical_features:
            '95in10': is category if 95% of the column fall into 10 values.
       - tex2int: if non-categorical text/string columns be mapped to integers
       - n_limit: max number of distinct values to one-hot encode,
         remaining values with fewer occurence are put in [colname]_other_ column.

    2. feed in data by set_training_data, then apply fit() function to tune the encoder.

    3. produce(): input data would be encoded and return.
    """

    def __repr__(self):
        return "%s(%r)" % ('Encoder', self.__dict__)


    def __init__(self, *, categorical_features='95in10', text2int=True, n_limit=10) -> None:

        self.categorical_features = categorical_features
        self.n_limit = n_limit
        self.text2int = text2int
        
        self.mapping = None
        self.all_columns = []
        self.empty_columns = []

        self.training_inputs = None
        self.fitted = False


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
                self.empty_columns.append(col.name)
                return

	    # if dtype = integer
            elif col.dtype.kind in np.typecodes['AllInteger']+'u':
                if isCat_95in10(col):
                    return self.__column_features(col.astype(str), n_limit)

            # if dtype = category
            elif col.dtype.name == 'category':
                return self.__column_features(col, n_limit)

            # for the rest other than float
            elif col.dtype.kind not in np.typecodes['AllFloat']:
                if isCat_95in10(col):
                    return self.__column_features(col, n_limit)

            return


    def get_params(self) -> Params:
        return Params(mapping=self.mapping, all_columns=self.all_columns, empty_columns=self.empty_columns)


    def set_params(self, *, params: Params) -> None:
        self.fitted = True
        self.mapping = params.mapping
        self.all_columns = params.all_columns
        self.empty_columns = params.empty_columns


    def set_training_data(self, *, inputs: Sequence[Input]):
        self.training_inputs = inputs
        self.fitted = False


    def fit(self, *, timeout:float = None, iterations: int = None) -> None:
        """
        Need training data from set_training_data first.
        The encoder would record categorical columns identified and
        the corresponding (with top n occurrence) column values to
        one-hot encode later in the produce step.
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        data_copy = self.training_inputs.copy()

        self.all_columns = set(data_copy.columns)

        if self.categorical_features == '95in10':
            idict = {}
            for column_name in data_copy:
                col = data_copy[column_name]
                p = self.__process(col, self.categorical_features, self.n_limit)
                if p:
                    idict[p[0]] = p[1]
            self.mapping = idict
        self.fitted = True


    def produce(self, *, inputs: Sequence[Input], timeout:float = None, iterations: int = None):
        """
        Convert and output the input data into encoded format,
        using the trained (fitted) encoder.
        Notice that a [colname]_other_ column is always kept for
        one-hot encoded columns.
        Missing(NaN) cells in a column one-hot encoded would give
        out a row of all-ZERO columns for the target column.
        """
        
        if isinstance(inputs, pd.DataFrame):
            data_copy = inputs.copy()
        else:
            data_copy = inputs[0].copy()

        data_enc = data_copy[list(self.mapping.keys())]
        data_else = data_copy.drop(self.mapping.keys(),axis=1)

        set_columns = set(data_copy.columns)

        if set_columns != self.all_columns:
            raise ValueError('Columns(features) fed at produce() differ from fitted data.')

        data_enc = data_copy[list(self.mapping.keys())]
        data_else = data_copy.drop(self.mapping.keys(),axis=1)

        res = []
        for column_name in data_enc:
            col = data_enc[column_name]
            col.is_copy = False

            chg_t = lambda x: str(int(x)) if type(x) is not str else x
            col[col.notnull()] = col[col.notnull()].apply(chg_t)
            
            chg_v = lambda x: 'other_' if (x and x not in self.mapping[col.name]) else x
            col = col.apply(chg_v)
            
            encoded = pd.get_dummies(col, dummy_na=True, prefix=col.name)

            missed = (["%s_%s"%(col.name,str(i)) for i in self.mapping[col.name] if
                    "%s_%s"%(col.name,str(i)) not in list(encoded.columns)])

            for m in missed:
                encoded[m] = 0

            res.append(encoded)

        data_else.drop(self.empty_columns, axis=1, inplace=True)
        if self.text2int:
            for column_name in data_else:
                if data_else[column_name].dtype.kind not in np.typecodes['AllInteger']+'uf':
                    data_else[column_name] = text2int(data_else[column_name])

        res.append(data_else)
        result = pd.concat(res, axis=1)

        return result


# example
if __name__ == '__main__':
    enc = Encoder()
    df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'B':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]})
    train_x = df
    enc.set_training_data(inputs=train_x)
    enc.fit()
    print(enc.produce(inputs=df))
    
    #save model for later use
    model = enc.get_params()

    enc2 = Encoder()
    df2 = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'B':[2.0,7,3,1,6,1,2,4,2,5,1,2,4,4,3]})
    enc2.set_params(params=model)
    print(enc2.produce(inputs=df2))
