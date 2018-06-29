import pandas as pd
import itertools
from itertools import groupby
from itertools import chain
import dateparser
import json

# add more later if needed
special_characters = ['-', '_', ' ']


class FoldColumns(object):
    def __init__(self, df, ignore_list=list()):
        self.df = df
        self.column_names = list(df)
        self.prefix_dict = self.create_prefix_dict(self.column_names)
        self.ignore_list = ignore_list

    @staticmethod
    def has_numbers(str):
        return any(char.isdigit() for char in str)

    @staticmethod
    def has_numbers_or_special_characters(str):
        return any(char.isdigit() or char in special_characters for char in str)

    def create_prefix_dict(self, column_names):
        """
        This function will create a dictionary with varying length prefixes of all column names
        :param column_names: the list of input column names
        :return: a dict: ['a','bc'] -> {'a':[0], 'b': [1], 'bc':[1]}
        """
        prefix_dict = {}
        for c in range(len(column_names)):
            column_name = column_names[c]
            # only take into consideration those columns which have a special character or numbers in it
            if self.has_numbers_or_special_characters(column_name):
                for i in range(len(column_name) + 1):
                    prefix = column_name[0:i]
                    if prefix != '' and not self.has_numbers(prefix) and len(prefix) > 1:
                        if column_name[0:i] not in prefix_dict:
                            prefix_dict[column_name[0:i]] = []
                        prefix_dict[column_name[0:i]].append(c)

        for key in list(prefix_dict):
            if len(prefix_dict[key]) <= 1:
                del prefix_dict[key]
        return prefix_dict

    def check_if_seen(self, str1, seen_prefix_dict):
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

        if len(list(set(self.prefix_dict[str1]) - set(column_indices_seen))) > 0:
            self.prefix_dict[str1] = list(set(self.prefix_dict[str1]) - set(column_indices_seen))
            return False
        return True

    @staticmethod
    def check_if_numbers_contiguous(indices_list):
        for i in range(len(indices_list) - 1):
            if indices_list[i] != (indices_list[i + 1] - 1):
                return False
        return True

    def detect(self):
        """
        call both prefix and date method and return  a combined list
        :return: list of list of columns to fold
        """
        prefix_lst = self.detect_columns_to_fold_prefix()
        dates_list = self.detect_columns_to_fold_dates()
        c_list = []
        if len(prefix_lst) > 0:
            c_list.extend(prefix_lst)

        if len(dates_list) > 0:
            c_list.extend(dates_list)
        return c_list

    def detect_columns_to_fold_prefix(self):
        sorted_prefix_lst = sorted(self.prefix_dict, key=len, reverse=True)
        valid_seen_prefixes = dict()

        for prefix in sorted_prefix_lst:
            if not self.check_if_seen(prefix, valid_seen_prefixes):
                if self.check_if_numbers_contiguous(self.prefix_dict[prefix]):
                    valid_seen_prefixes[prefix] = self.prefix_dict[prefix]

        for p in list(self.prefix_dict):
            if p not in valid_seen_prefixes:
                del self.prefix_dict[p]

        columns_list = []
        for key in list(self.prefix_dict):
            columns_list.extend(self.prefix_dict[key])

        return [self.prefix_dict[key] for key in list(self.prefix_dict)]

    def detect_columns_to_fold_dates(self):
        """
        This function will return a list of column names if they are dates
        :return: a subset of column_names
        """
        result = list()
        for index in range(len(self.column_names)):
            column_name = self.column_names[index]
            # do not want 12 to be parsed as date, minimum length should be 4 (year in YYYY format)
            if len(column_name) >= 4:
                try:
                    # for now strict parsing is true, otherwise it'll parse 'year' as valid date.
                    # in future, we'll have to specify date formats
                    parsed_column_as_date = dateparser.parse(column_name, settings={'STRICT_PARSING': True})
                    if parsed_column_as_date:
                        # column_name has been parsed as a valid date, it is a candidate for fold
                        result.append(i)
                except:
                    # something went wrong, doesn't matter what
                    pass
        return result

    def perform(self, columns_list_to_fold):
        df = None
        for columns_to_fold in columns_list_to_fold:
            df = self.fold_columns(columns_to_fold)
        return df if df is not None else self.df

    def fold_columns(self, columns_to_fold_all):
        columns_to_fold = list(set(columns_to_fold_all) - set(self.ignore_list))
        if len(columns_to_fold) == 0:
            # nothing to fold, return the original
            return df
        new_column_suffix = ''
        for c in columns_to_fold:
            new_column_suffix += '_' + str(c)
        new_column_name = '{}_{}'.format(self.df.columns[columns_to_fold[0]], new_column_suffix)
        new_rows_list = list()
        orig_columns = list(range(len(self.column_names)))
        # subtract ignore list from columns_to_fold

        non_foldable_columns = list(set(orig_columns) - set(columns_to_fold))

        for i in df.index.values:
            row = df.iloc[i]
            for column_to_fold in columns_to_fold:

                d1 = {}
                for nfc in non_foldable_columns:
                    d1[self.column_names[nfc]] = row[self.column_names[nfc]]

                d1[new_column_name] = self.column_names[column_to_fold]
                d1['{}_value'.format(new_column_name)] = row[column_to_fold]
                new_rows_list.append(d1)

        new_df = pd.DataFrame(new_rows_list)
        return new_df


if __name__ == '__main__':
    # input_file = '/tmp/sample.csv'
    input_file = 'data_sample/sample1.csv'
    # print(check_if_substring('dft', ['sfsdft', 'serers', 'sssss', 'dfe']))
    df = pd.read_csv(input_file)
    # process_column_prefix(df)
    fc = FoldColumns(df, ignore_list=[])
    c_list = fc.detect()
    print(fc.perform(c_list))
    # f = detect_columns_to_fold(df)
