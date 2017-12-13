import pandas as pd
import numpy as np
from typing import NamedTuple, Sequence
import copy
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Input = pd.DataFrame
Output = pd.DataFrame

Params = NamedTuple('Params', [
    ('mapping', dict),
    ('all_columns', list),
    ('empty_columns', list),
    ('textmapping', dict),
    ('target_columns',dict)
    ])

class UnaryEncoder(UnsupervisedLearnerPrimitiveBase[Input, Output, Params]):

    def __repr__(self):
        return "%s(%r)" % ('UnaryEncoder', self.__dict__)


    def __init__(self, *, text2int=False) -> None:
        self.text2int = text2int
        self.textmapping = None
        self.mapping = None
        self.all_columns = []
        self.empty_columns = []

        self.training_inputs = None
        self.target_columns = None
        self.fitted = False


    def get_params(self) -> Params:
        return Params(mapping=self.mapping, all_columns=self.all_columns, empty_columns=self.empty_columns,
                      textmapping=self.textmapping, target_columns = self.target_columns)


    def set_params(self, *, params: Params) -> None:
        self.fitted = True
        self.mapping = params.mapping
        self.all_columns = params.all_columns
        self.empty_columns = params.empty_columns
        self.textmapping = params.textmapping
        self.target_columns = params.target_columns


    def set_training_data(self, *, inputs: Sequence[Input], targets: list) -> None:
        self.training_inputs = inputs
        self.target_columns = targets
        self.fitted = False


    def fit(self, *, timeout:float = None, iterations: int = None) -> None:
        """
        Need training data from set_training_data first.
        The encoder would record specified columns to encode and column values to
        unary encode later in the produce step.
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        data_copy = self.training_inputs.copy()

        self.all_columns = set(data_copy.columns)

        # mapping
        if not set(self.target_columns).issubset(set(data_copy.columns)):
            raise ValueError('Target columns are not subset of columns in training_inputs.')

        idict = {}
        for target_name in self.target_columns:
            col = data_copy[target_name]
            idict[target_name] = sorted(col.unique())
        self.mapping = idict

        if self.text2int:
            texts = data_copy.drop(self.mapping.keys(),axis=1)
            texts = texts.select_dtypes(include=[object])
            le = Label_encoder()
            le.fit_pd(texts)
            self.textmapping = le.get_params()

        self.fitted = True


    def __encode_column(self, col):
        unary = pd.DataFrame(col)
        for v in self.mapping[col.name]:
            unary[col.name+"_"+str(v)] = (col >= v).astype(int)
        return unary.drop(col.name,axis=1)


    def produce(self, *, inputs: Sequence[Input], timeout:float = None, iterations: int = None) -> pd.DataFrame:
        """
        Convert and output the input data into unary encoded format,
        using the trained (fitted) encoder.
        Value unseen in training_inputs would be rounded to nearest value in training_inputs.
        Missing(NaN) cells in a column one-hot encoded would give
        out a row of all-ZERO columns for the target column.
        """
        if not self.fitted:
            raise ValueError('Encoder model not fitted. Use fit()')

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

            #chg_t = lambda x: str(int(x)) if type(x) is not str else x
            #col[col.notnull()] = col[col.notnull()].apply(chg_t)

            chg_v = lambda x: min(self.mapping[col.name], key=lambda a:abs(a-x)) if x is not None else x
            col[col.notnull()] = col[col.notnull()].apply(chg_v)
            encoded = self.__encode_column(col)
            res.append(encoded)

        data_else.drop(self.empty_columns, axis=1, inplace=True)
        if self.text2int:
            texts = data_else.select_dtypes([object])
            le = Label_encoder()
            le.set_params(self.textmapping)
            data_else[texts.columns] = le.transform_pd(texts)

        res.append(data_else)
        result = pd.concat(res, axis=1)

        return result
