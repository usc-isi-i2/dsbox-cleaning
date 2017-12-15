import numpy as np
import pandas as pd
import copy

from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import NamedTuple, Dict, List, Set
from d3m_metadata.container.pandas import DataFrame
from d3m_metadata.hyperparams import Enumeration, UniformInt, Hyperparams

def isCat_95in10(col):
    """
    hardcoded rule for identifying (integer/string) categorical column
    """
    return col.value_counts().head(10).sum() / float(col.count()) > .95

Input = DataFrame
Output = DataFrame

Params = NamedTuple('Params', [
    ('mapping', Dict),
    ('all_columns', Set[str]),
    ('empty_columns', List[str]),
    ('textmapping', Dict)
    ])


class EncHyperparameter(Hyperparams):
    text2int = Enumeration(values=[True,False],default=False,
            description='Whether to convert everything to numerical')
    n_limit = UniformInt(lower=5, upper=100, default=12,description='Maximum columns to encode')


## reference: https://github.com/scikit-learn/scikit-learn/issues/8136
class Label_encoder(object):
    def __init__(self):
        self.class_index = None

    def fit_pd(self, df, cols=[]):
        '''
        fit all columns in the df or specific list.
        generate a dict:
        {feature1:{label1:1,label2:2}, feature2:{label1:1,label2:2}...}
        '''
        if len(cols) == 0:
            cols = df.columns
        self.class_index = {}
        for f in cols:
            uf = df[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1

    def transform_pd(self,df,cols=[]):
        '''
        transform all columns in the df or specific list from lable to index, return an update dataframe.
        '''
        newdf = copy.deepcopy(df)
        if len(cols) == 0:
            cols = df.columns
        for f in cols:
            if f in self.class_index:
                newdf[f] = df[f].apply(lambda d: self.__update_label(f,d))
        return newdf

    def get_params(self):
        return self.class_index

    def set_params(self, textmapping):
        self.class_index = textmapping

    def __update_label(self,f,x):
        '''
        update the label to index, if not found in the dict, add and update the dict.
        '''
        try:
            return self.class_index[f][x]
        except:
            self.class_index[f][x] = max(self.class_index[f].values())+1
            return self.class_index[f][x]


class Encoder(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, EncHyperparameter]):
    __author__ = "USC ISI"
    __metadata__ = {
            "id": "18f0bb42-6350-3753-8f2d-d1c3da70f279",
            "name": "dsbox.datapreprocessing.cleaner.Encoder",
            "common_name": "DSBox Data Encoder",
            "team": "USC ISI",
            "task_type": [ "Data preprocessing" ],
            "tags": [ "preprocesing", "encoding" ],
            "description": "Encode data, such as one-hot encoding for categorical data",
            "languages": [ "python3.5", "python3.6" ],
            "library": "dsbox",
            "version": "0.2.0",
            "is_class": True,
            "schema_version": 1.0,
            "parameters": [
                {
                    "name": "n_limit",
                    "description": "Max number of distinct values to one-hot encode",
                    "type": "int",
                    "default": "10",
                    "is_hyperparameter": True
                    },
                {
                    "name": "text2int",
                    "description": "Encode text strings as integers. For example, set to true if text strings encode categorical data, and set to false if text strings contain natural language text.",
                    "type": "boolean",
                    "default": "true",
                    "is_hyperparameter": True
                    }
                ],
            "build": [ { "type": "pip", "package": "dsbox-datacleaning" } ],
            "compute_resources": {
                "cores_per_node": [],
                "disk_per_node": [],
                "expected_running_time": [],
                "gpus_per_node": [],
                "mem_per_gpu": [],
                "mem_per_node": [],
                "num_nodes": [],
                "sample_size":[],
                "sample_unit":[]
                },
            "interfaces": [ "UnsupervisedLearnerPrimitiveBase" ],
            "interfaces_version": "2017.9.22rc0",

            }

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


    def __init__(self, hyperparam: EncHyperparameter, categorical_features='95in10') -> None:

        self.categorical_features = categorical_features
        self.n_limit = hyperparam['n_limit']
        self.text2int = hyperparam['text2int']
        #
        self.textmapping : Dict = None
        #
        self.mapping : Dict = None
        self.all_columns : Set[str] = set()
        self.empty_columns : List[str] = []

        self.training_inputs : Input = None
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
        return Params(mapping=self.mapping, all_columns=self.all_columns, empty_columns=self.empty_columns, textmapping=self.textmapping)


    def set_params(self, *, params: Params) -> None:
        self.fitted = True
        self.mapping = params.mapping
        self.all_columns = params.all_columns
        self.empty_columns = params.empty_columns
        self.textmapping = params.textmapping

    def set_training_data(self, *, inputs: Input):
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
            #
            if self.text2int:
                texts = data_copy.drop(self.mapping.keys(),axis=1)
                texts = texts.select_dtypes(include=[object])
                le = Label_encoder()
                le.fit_pd(texts)
                self.textmapping = le.get_params()
            #
        self.fitted = True


    def produce(self, *, inputs: Input, timeout:float = None, iterations: int = None):
        """
        Convert and output the input data into encoded format,
        using the trained (fitted) encoder.
        Notice that [colname]_other_ and [colname]_nan columns 
        are always kept for one-hot encoded columns.
        """

        if isinstance(inputs, pd.DataFrame):
            data_copy = inputs.copy()
        else:
            data_copy = inputs[0].copy()
 
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
            texts = data_else.select_dtypes([object])
            le = Label_encoder()
            le.set_params(self.textmapping)
            data_else[texts.columns] = le.transform_pd(texts)
            #for column_name in data_else:
            #    if data_else[column_name].dtype.kind not in np.typecodes['AllInteger']+'uf':
            #        data_else[column_name] = text2int(data_else[column_name])
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
