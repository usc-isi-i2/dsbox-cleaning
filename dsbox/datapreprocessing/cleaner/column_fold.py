import pandas as pd
import dateparser
import numpy as np
from d3m import container
import d3m.metadata.base as mbase


from typing import Dict

from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from . import config

# add more later if needed

special_characters = ['-', '_', ' ']

Input = container.DataFrame
Output = container.DataFrame


class FoldParams(params.Params):
    mapping: Dict


class FoldHyperparameter(hyperparams.Hyperparams):
    pass


class FoldColumns(UnsupervisedLearnerPrimitiveBase[Input, Output, FoldParams, FoldHyperparameter]):

    # TODO update metadata
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "dsbox-fold-columns",
        "version": config.VERSION,
        "name": "DSBox Fold Columns",
        "description": "Fold Columns",
        "python_path": "d3m.primitives.dsbox.FoldColumns",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": ["fold"],
        "installation": [config.INSTALLATION],
        "location_uris": [],
        "hyperparms_to_tune": []
    })

    def __repr__(self):
        return "%s(%r)" % ('Column_folder', self.__dict__)

    def __init__(self, *, hyperparams: FoldHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        self._df: Input = None
        self._ignore_list = list()
        self._mapping: Dict = {}
        self._fitted = False

    def set_training_data(self, *, inputs: Input) -> None:
        self._df = inputs
        self._fitted = False

    def get_params(self) -> FoldParams:
        if not self._fitted:
            raise ValueError("FoldColumns: Fit not performed.")
        return FoldParams(mapping=self._mapping)

    def set_params(self, *, params: FoldParams) -> None:
        self._fitted = True
        self._mapping = params['mapping']

    @staticmethod
    def _has_numbers(str):
        return any(char.isdigit() for char in str)

    @staticmethod
    def _has_numbers_or_special_characters(str):
        return any(char.isdigit() or char in special_characters for char in str)

    def _create_prefix_dict(self, column_names):
        """
        This function will create a dictionary with varying length prefixes of all column names
        :param column_names: the list of input column names
        :return: a dict: ['a','bc'] -> {'a':[0], 'b': [1], 'bc':[1]}
        """
        prefix_dict = {}
        for c in range(len(column_names)):
            column_name = column_names[c]
            # only take into consideration those columns which have a special character or numbers in it
            if self._has_numbers_or_special_characters(column_name):
                for i in range(len(column_name) + 1):
                    prefix = column_name[0:i]
                    if prefix != '' and not self._has_numbers(prefix) and len(prefix) > 1:
                        if column_name[0:i] not in prefix_dict:
                            prefix_dict[column_name[0:i]] = []
                        prefix_dict[column_name[0:i]].append(c)

        for key in list(prefix_dict):
            if len(prefix_dict[key]) <= 1:
                del prefix_dict[key]
        return prefix_dict

    def _check_if_seen(self, str1, seen_prefix_dict):
        """
        This function returns true if str1 is a subtring of any string in str_list and that string startswith str1
        :param str1: string to be checked
        :param str_lst: input list of strings
        :return: True if str1 is a substring of any of the strings else False
        """
        column_indices_seen = []
        for str in list(seen_prefix_dict):
            if str.startswith(str1):
                column_indices_seen.extend(seen_prefix_dict[str])

        if len(column_indices_seen) == 0:
            return False

        if len(list(set(self._prefix_dict[str1]) - set(column_indices_seen))) > 0:
            self._prefix_dict[str1] = list(set(self._prefix_dict[str1]) - set(column_indices_seen))
            return False
        return True

    @staticmethod
    def _check_if_numbers_contiguous(indices_list):
        for i in range(len(indices_list) - 1):
            if indices_list[i] != (indices_list[i + 1] - 1):
                return False
        return True

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        call both prefix and date method and return  a combined list
        :return: list of list of columns to fold
        """
        if self._fitted:
            return

        if self._df is None:
            raise ValueError('Missing training(fitting) data.')

        self._column_names = list(self._df) if self._df is not None else []
        self._prefix_dict = self._create_prefix_dict(self._column_names)
        prefix_lst = self._detect_columns_to_fold_prefix()
        dates_list = self._detect_columns_to_fold_dates()
        c_list = []
        if len(prefix_lst) > 0:
            c_list.extend(prefix_lst)

        if len(dates_list) > 0:
            c_list.extend(dates_list)
        print(c_list)

        self._mapping = {'foldable_columns': c_list}
        self._fitted = True

    def _detect_columns_to_fold_prefix(self):
        sorted_prefix_lst = sorted(self._prefix_dict, key=len, reverse=True)
        valid_seen_prefixes = dict()

        for prefix in sorted_prefix_lst:
            if not self._check_if_seen(prefix, valid_seen_prefixes):
                if self._check_if_numbers_contiguous(self._prefix_dict[prefix]):
                    valid_seen_prefixes[prefix] = self._prefix_dict[prefix]

        for p in list(self._prefix_dict):
            if p not in valid_seen_prefixes:
                del self._prefix_dict[p]

        columns_list = []
        for key in list(self._prefix_dict):
            columns_list.extend(self._prefix_dict[key])

        return [self._prefix_dict[key] for key in list(self._prefix_dict)]

    def _detect_columns_to_fold_dates(self):
        """
        This function will return a list of column names if they are dates
        :return: a subset of column_names
        """
        result = list()
        for index in range(len(self._column_names)):
            column_name = self._column_names[index]
            # do not want 12 to be parsed as date, minimum length should be 4 (year in YYYY format)
            if len(column_name) >= 4:
                try:
                    # for now strict parsing is true, otherwise it'll parse 'year' as valid date.
                    # in future, we'll have to specify date formats
                    parsed_column_as_date = dateparser.parse(column_name, settings={'STRICT_PARSING': True})
                    if parsed_column_as_date:
                        # column_name has been parsed as a valid date, it is a candidate for fold
                        result.append(index)
                except:
                    # something went wrong, doesn't matter what
                    pass
        return result

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        columns_list_to_fold = self._mapping.get('foldable_columns', [])
        if len(columns_list_to_fold) == 0:
            return CallResult(self._df, True, 1)

        df = None
        for columns_to_fold in columns_list_to_fold:
            df = self._fold_columns(columns_to_fold)
        df.to_csv("/Users/runqishao/Desktop/aaa.csv")
        return CallResult(df, True, 1) if df is not None else CallResult(self._df, True, 1)

    def _fold_columns(self, columns_to_fold_all):
        columns_to_fold = list(set(columns_to_fold_all) - set(self._ignore_list))
        if len(columns_to_fold) == 0:
            # nothing to fold, return the original
            return self._df
        new_column_suffix = ''
        for c in columns_to_fold:
            new_column_suffix += '_' + str(c)
        new_column_name = '{}_{}'.format(self._df.columns[columns_to_fold[0]], new_column_suffix)
        new_rows_list = list()
        orig_columns = list(range(len(self._column_names)))
        # subtract ignore list from columns_to_fold

        non_foldable_columns = list(set(orig_columns) - set(columns_to_fold))

        for i in self._df.index.values:
            row = self._df.iloc[i]
            for column_to_fold in columns_to_fold:

                d1 = {}
                for nfc in non_foldable_columns:
                    d1[self._column_names[nfc]] = row[self._column_names[nfc]]

                d1[new_column_name] = self._column_names[column_to_fold]
                d1['{}_value'.format(new_column_name)] = row[column_to_fold]
                new_rows_list.append(d1)

        new_df = pd.DataFrame(new_rows_list)
        return new_df
