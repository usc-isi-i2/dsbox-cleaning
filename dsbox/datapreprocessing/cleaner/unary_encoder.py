import pandas as pd  # type: ignore
import copy
import typing

from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m_metadata import hyperparams, container, params
from d3m_metadata.metadata import PrimitiveMetadata

#from dsbox.datapreprocessing.profiler import category_detection

Input = container.DataFrame
Output = container.DataFrame

#def __default_targets(data):
#    is_category = category_detection.category_detect(data)
#    res = []
#    cnt = -1
#    for col_name in data:
#        cnt += 1
#        if col.dtype in [int, np.int64, np.int32, np.int16, np.int8] and is_category[col_name]:
#            res.append(cnt)

class Params(params.Params):
    mapping : typing.Dict
    all_columns : typing.Set[str]
    empty_columns : typing.List
    textmapping : typing.Dict
    target_columns : typing.List[int]


class UEncHyperparameter(hyperparams.Hyperparams):
    text2int = hyperparams.Enumeration(values=[True,False],default=False, 
            description='Whether to convert everything to numerical')
    targetColumns= hyperparams.Hyperparameter(
        default= [0],
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
        description="The index of the column to be encoded")

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
        transform all columns in the df or specific list from lable to index, return and update dataframe.
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


class UnaryEncoder(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, UEncHyperparameter]):
    
    metadata = PrimitiveMetadata({
        "id": "DSBox-unary-encoder",
        "version": "0.3.1",
        "name": "DSBox Unary Data Encoder",
        "description": "Encode using unary code for orinal data",
        "python_path": "d3m.primitives.dsbox.Encoder",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": [ "ENCODE_ONE_HOT" ],  # !!!! Need to submit algorithm type "Imputation"
        "source": {
            "name": "USC ISI",
            "uris": [
                "https://github.com/usc-isi-i2/dsbox-cleaning.git"
                ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "preprocessing",  "encoding"],
        "installation": [ 
            {
                "type": "PIP",
                "package": "dsbox-datacleaning",
                "version": "0.3.1" 
            } 
        ],
        #"location_uris": [],
        #"precondition": [],
        #"effects": [],
        #"hyperparms_to_tune": []
        })    

    def __repr__(self):
        return "%s(%r)" % ('UnaryEncoder', self.__dict__)


    def __init__(self, *, hyperparams: UEncHyperparameter, random_seed: int = 0, 
                 docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers
        self._text2int = hyperparams['text2int']
        self._target_columns = hyperparams['targetColumns']
        self._textmapping = dict()
        self._mapping = dict()
        self._all_columns = set()
        self._empty_columns = []

        self._training_inputs = None
        self._fitted = False


    def get_params(self) -> Params:
        
        # Hack to work around pytypes bug. Covert numpy int64 to int. 
        for key in self._mapping.keys():
            self._mapping[key] = [int(x) for x in self._mapping[key]]

        param = Params(mapping=self._mapping, all_columns=self._all_columns, empty_columns=self._empty_columns,
                       textmapping=self._textmapping, target_columns = self._target_columns)
        return param


    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._mapping = params['mapping']
        self._all_columns = params['all_columns']
        self._empty_columns = params['empty_columns']
        self._textmapping = params['textmapping']
        self._target_columns = params['target_columns']


    def set_training_data(self, *, inputs: Input) -> None:
        self._training_inputs = inputs
        self._fitted = False


    def fit(self, *, timeout:float = None, iterations: int = None) -> None:
        """
        Need training data from set_training_data first.
        The encoder would record specified columns to encode and column values to
        unary encode later in the produce step.
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        data_copy = self._training_inputs.copy()

        self._all_columns = set(data_copy.columns)

        # mapping
        if max(self._target_columns) > len(data_copy.columns)-1:
            raise ValueError('Target columns are not subset of columns in training_inputs.(Out of range).')

        idict = {}
        for target_id in self._target_columns:
            name = data_copy.columns[target_id]
            col = data_copy[name]
            idict[name] = sorted(col.unique())
        self._mapping = idict

        if self._text2int:
            texts = data_copy.drop(self._mapping.keys(),axis=1)
            texts = texts.select_dtypes(include=[object])
            le = Label_encoder()
            le.fit_pd(texts)
            self._textmapping = le.get_params()

        self._fitted = True


    def __encode_column(self, col):
        unary = pd.DataFrame(col)
        for v in self._mapping[col.name]:
            unary[col.name+"_"+str(v)] = (col >= v).astype(int)
        return unary.drop(col.name,axis=1)


    def produce(self, *, inputs: Input, timeout:float = None, iterations: int = None) -> pd.DataFrame:
        """
        Convert and output the input data into unary encoded format,
        using the trained (fitted) encoder.
        Value unseen in training_inputs would be rounded to nearest value in training_inputs.
        Missing(NaN) cells in a column one-hot encoded would give
        out a row of all-ZERO columns for the target column.
        """
        if not self._fitted:
            raise ValueError('Encoder model not fitted. Use fit()')

        if isinstance(inputs, pd.DataFrame):
            data_copy = inputs.copy()
        else:
            data_copy = inputs[0].copy()

        set_columns = set(data_copy.columns)

        if set_columns != self._all_columns:
            raise ValueError('Columns(features) fed at produce() differ from fitted data.')

        data_enc = data_copy[list(self._mapping.keys())]
        data_else = data_copy.drop(self._mapping.keys(),axis=1)

        res = []
        for column_name in data_enc:
            col = data_enc[column_name]
            col.is_copy = False

            chg_v = lambda x: min(self._mapping[col.name], key=lambda a:abs(a-x)) if x is not None else x
            col[col.notnull()] = col[col.notnull()].apply(chg_v)
            encoded = self.__encode_column(col)
            res.append(encoded)

        data_else.drop(self._empty_columns, axis=1, inplace=True)
        if self._text2int:
            texts = data_else.select_dtypes([object])
            le = Label_encoder()
            le.set_params(self._textmapping)
            data_else[texts.columns] = le.transform_pd(texts)

        res.append(data_else)
        result = pd.concat(res, axis=1)

        return result
