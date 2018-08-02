import re
from dateutil.parser import parse
import d3m.metadata.base as mbase
import pandas as pd

"""
this script contains all the helper functions that apply to a input string
most refer from: https://github.com/usc-isi-i2/dptk

"""
UNIQUE_VALUE_TO_BE_CATEGORICAL = 20
RATIO_TO_BE_CATEGORICAL = 0.3

NEGATIVE_SEMANTIC_TYPES = set(["https://metadata.datadrivendiscovery.org/types/FileName",
                               "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                               "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                               "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                               "https://metadata.datadrivendiscovery.org/types/UniqueKey",
                               "https://metadata.datadrivendiscovery.org/types/Location",
                               "https://metadata.datadrivendiscovery.org/types/DatasetResource",
                               "http://schema.org/Boolean",
                               "http://schema.org/Integer",
                               "http://schema.org/Float",
                               "https://metadata.datadrivendiscovery.org/types/FilesCollection",
                               "https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint"])


class HelperFunction:

    @staticmethod
    def convertAlphatoNum(input):

        non_decimal = re.compile(r'[^\d\.]+')
        return non_decimal.sub(' ', input)

    @staticmethod
    def is_Integer_Number_Ext(s):
        """
            return any(char.isdigit() for char in inputString)
        """
        try:
            int(s)
            return True
        except:
            try:
                int(HelperFunction.convertAlphatoNum(s))
                return True
            except:
                return False

    @staticmethod
    def is_Decimal_Number_Ext(s):
        try:
            float(s)
            return True
        except:
            try:
                float(HelperFunction.convertAlphatoNum(s))
                return True
            except:
                return False

    @staticmethod
    def is_Integer_Number(s):
        # return any(char.isdigit() for char in inputString)
        try:
            int(s)
            return True
        except:
            return False

    @staticmethod
    def is_Decimal_Number(s):
        try:
            float(s)
            return True
        except:
            return False

    @staticmethod
    def is_date(string):
        try:
            parse(string)
            return True
        except ValueError:
            return False

    @staticmethod
    def getDecimal(s):
        try:
            return float(s)
        except:
            try:
                return float(HelperFunction.convertAlphatoNum(s))
            except:
                return 0.0

    @staticmethod
    def is_categorical(df_col):
        uniques = df_col.unique()
        length = df_col.shape[0]
        if len(uniques) <= UNIQUE_VALUE_TO_BE_CATEGORICAL and len(uniques) / length < RATIO_TO_BE_CATEGORICAL:
            return True
        return False

    @staticmethod
    def cols_to_clean(sampled_df, POSITIVE_SEMANTIC_TYPES):
        cols = list()
        for col_idx in range(len(sampled_df.columns)):
            semantic_types = list(
                dict(sampled_df.metadata.query((mbase.ALL_ELEMENTS, col_idx))).get("semantic_types", tuple([])))

            if "https://metadata.datadrivendiscovery.org/types/FloatVector" in semantic_types:
                cols.append(col_idx)
            elif POSITIVE_SEMANTIC_TYPES.intersection(semantic_types) and not NEGATIVE_SEMANTIC_TYPES.intersection(
                    semantic_types):
                cols.append(col_idx)

        return cols

    @staticmethod
    def custom_is_null(x, column_dtype=object):
        if column_dtype == object and x == "":
            return True
        elif pd.isnull(x):
            return True
        return False
