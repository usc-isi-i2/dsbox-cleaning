import re
from dateutil.parser import parse

"""
this script contains all the helper functions that apply to a input string
most refer from: https://github.com/usc-isi-i2/dptk

"""
UNIQUE_VALUE_TO_BE_CATEGORICAL = 20
RATIO_TO_BE_CATEGORICAL = 0.3


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
