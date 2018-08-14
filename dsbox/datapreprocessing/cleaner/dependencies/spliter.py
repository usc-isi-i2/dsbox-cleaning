#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import d3m.metadata.base as mbase

from common_primitives import utils
from d3m.container import DataFrame as d3m_DataFrame
from dsbox.datapreprocessing.cleaner.dependencies.helper_funcs import HelperFunction

AVG_LENGTH_MAX = 30


def update_type(extends, df_origin):
    extends_df = d3m_DataFrame.from_dict(extends)
    if extends != {}:
        extends_df.index = df_origin.index.copy()
    new_df = utils.append_columns(df_origin, extends_df)

    indices = list()
    for key in extends:
        indices.append(new_df.columns.get_loc(key))

    for idx in indices:
        old_metadata = dict(new_df.metadata.query((mbase.ALL_ELEMENTS, idx)))

        numerics = pd.to_numeric(new_df.iloc[:, idx], errors='coerce')
        length = numerics.shape[0]
        nans = numerics.isnull().sum()

        if nans / length > 0.9:
            if HelperFunction.is_categorical(new_df.iloc[:, idx]):
                old_metadata['semantic_types'] = (
                    "https://metadata.datadrivendiscovery.org/types/CategoricalData",)
            else:
                old_metadata['semantic_types'] = ("http://schema.org/Text",)
        else:
            intcheck = (numerics % 1) == 0
            if np.sum(intcheck) / length > 0.9:
                old_metadata['semantic_types'] = ("http://schema.org/Integer",)
            else:
                old_metadata['semantic_types'] = ("http://schema.org/Float",)

        old_metadata['semantic_types'] += ("https://metadata.datadrivendiscovery.org/types/Attribute",)

        new_df.metadata = new_df.metadata.update((mbase.ALL_ELEMENTS, idx), old_metadata)

    return new_df


class PhoneParser:

    @staticmethod
    def detect(df, columns_ignore=list()):
        positive_semantic_types = set(["http://schema.org/Text"])

        cols_to_detect = HelperFunction.cols_to_clean(df, positive_semantic_types)
        require_checking = \
            list(set(cols_to_detect).difference(set(columns_ignore)))
        extends = {"columns_to_perform": [], "split_to": []}
        for one_column in require_checking:
            if PhoneParser.is_phone(df.iloc[:, one_column]):
                extends["columns_to_perform"].append(one_column)
        return extends

    @staticmethod
    def perform(df, columns_perform):
        extends = {}
        for one_column in columns_perform["columns_to_perform"]:
            result = PhoneParser.phone_parser(df.iloc[:, one_column])
            try:
                extends[str(df.columns[one_column]) + '_phone'] = result
            except:
                extends[df.columns[one_column].apply(str) + '_phone'] = result

        new_df = update_type(extends, df)

        return new_df

    @staticmethod
    def is_phone(rows):
        pattern = \
            '^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'
        match_count = 0
        for row in rows:
            if re.match(pattern, str(row)):
                match_count += 1
        if float(match_count) / len(rows) > 0.5:
            return True
        return False

    @staticmethod
    def phone_parser(rows):
        pattern = \
            '^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'
        new_rows = []
        for row in rows:
            phone_match = re.match(pattern, str(row))
            number = ''
            group_id = 1
            if phone_match:
                number = ''
                while group_id < 5:
                    if phone_match.group(group_id):
                        number += phone_match.group(group_id) + '-'
                    group_id += 1
            number = number.strip('-')
            new_rows.append(number)
        return new_rows


class PunctuationParser:

    @staticmethod
    def detect(df, columns_ignore=list()):
        positive_semantic_types = set(["http://schema.org/Text"])
        cols_to_detect = HelperFunction.cols_to_clean(df, positive_semantic_types)
        require_checking = list(set(cols_to_detect).difference(set(columns_ignore)))
        extends = {"columns_to_perform": [], "split_to": []}
        for one_column in require_checking:
            rows = df.iloc[:, one_column]
            filtered_rows = [len(str(row)) for row in rows if len(str(row)) > 0]
            if len(filtered_rows) > 0:
                avg_len = sum(filtered_rows) / len(filtered_rows)
                if avg_len < AVG_LENGTH_MAX:
                    if not PunctuationParser.num_check(df.iloc[:, one_column]):
                        common_list = PunctuationParser.find_common(df.iloc[:, one_column])
                        if len(common_list) > 0:
                            result = PunctuationParser.splitter(df.iloc[:, one_column], common_list)
                            extends["columns_to_perform"].append(one_column)
                            extends["split_to"].append(len(result))
        return extends

    @staticmethod
    def perform(df, columns_perform):
        extends = {}
        for i, one_column in enumerate(columns_perform["columns_to_perform"]):
            common_list = PunctuationParser.find_common(df.iloc[:, one_column])
            result = PunctuationParser.splitter(df.iloc[:, one_column], common_list)
            if len(result) > columns_perform["split_to"][i]:
                result = result[:columns_perform["split_to"][i]]
            elif len(result) < columns_perform["split_to"][i]:
                for j in range(columns_perform["split_to"][i] - len(result)):
                    extra_column = np.reshape(np.asarray([np.nan] * len(result[0])), (1, len(result[0])))
                    result = np.append(result, extra_column, axis=0)
            count = 0
            for one in result:
                try:
                    extends[str(df.columns[one_column]) + '_punc_' + str(count)] = one
                except:
                    extends[df.columns[one_column].apply(str) + '_punc_' + str(count)] = one
                count += 1

        new_df = update_type(extends, df)

        return new_df

    @staticmethod
    def splitter(rows, common_list):
        new_rows = []
        max_column_num = 0
        constraints = [
            '^',
            '$',
            '\\',
            '|',
            '{',
            '[',
            '(',
            '*',
            '+',
            '?',
        ]
        re_list = ''
        for one_split in common_list:
            if one_split in constraints:
                re_list += '\\' + one_split + '|'
            else:
                re_list += one_split + '|'
        re_list.strip('|')
        for row in rows:
            new_row = [x for x in re.split(re_list, str(row)) if x]
            max_column_num = max(max_column_num, len(new_row))
            new_rows.append(new_row)

        row_count = 0
        while row_count < len(rows):
            if len(new_rows[row_count]) < max_column_num:
                new_rows[row_count].extend([np.nan] * (max_column_num - len(new_rows[row_count])))
            row_count += 1
        new_rows = np.array(new_rows).T
        return new_rows

    @staticmethod
    def num_check(rows, num_threshold=0.1):
        num_count = 0
        for row in rows:
            try:
                float(row)
                num_count += 1
                pass
            except Exception:
                pass
        if float(num_count) / len(rows) >= num_threshold:
            return True
        else:
            return False

    @staticmethod
    def find_common(rows, common_threshold=0.9):
        common_list = []
        appear_dict = {}
        for row in rows:
            for ch in str(row):
                if not (ch.isdigit() or ch.isalpha() or ch == '.'):
                    if ch in appear_dict:
                        appear_dict[ch] += 1
                    else:
                        appear_dict[ch] = 1
        for key in appear_dict:
            if float(appear_dict[key]) / len(rows) >= common_threshold:
                common_list.append(key)
        return common_list


class NumAlphaParser:

    @staticmethod
    def detect(df, columns_ignore=list()):
        positive_semantic_types = set(["http://schema.org/Text"])
        cols_to_detect = HelperFunction.cols_to_clean(df, positive_semantic_types)
        require_checking = list(set(cols_to_detect).difference(set(columns_ignore)))
        extends = {"columns_to_perform": [], "split_to": []}
        for one_column in require_checking:
            rows = df.iloc[:, one_column]
            filtered_rows = [len(str(row)) for row in rows if len(str(row)) > 0]
            if len(filtered_rows) > 0:
                avg_len = sum(filtered_rows) / len(filtered_rows)
                if avg_len < AVG_LENGTH_MAX:
                    if not NumAlphaParser.num_check(df.iloc[:, one_column]):
                        isnum_alpha = NumAlphaParser.is_num_alpha(df.iloc[:, one_column])
                        if isnum_alpha:
                            result = NumAlphaParser.num_alpha_splitter(df.iloc[:, one_column])
                            extends["columns_to_perform"].append(one_column)
                            extends["split_to"].append(len(result))

        return extends

    @staticmethod
    def perform(df, columns_perform):
        extends = {}
        for i, one_column in enumerate(columns_perform["columns_to_perform"]):
            result = NumAlphaParser.num_alpha_splitter(df.iloc[:, one_column])
            if len(result) > columns_perform["split_to"][i]:
                result = result[:columns_perform["split_to"][i]]
            elif len(result) < columns_perform["split_to"][i]:
                for j in range(columns_perform["split_to"][i] - len(result)):
                    result.append([np.nan] * len(result[0]))
            count = 0
            for one in result:
                try:
                    extends[str(df.columns[one_column]) + '_na_' + str(count)] = one
                except:
                    extends[df.columns[one_column].apply(str) + '_na_' + str(count)] = one
                count += 1

        new_df = update_type(extends, df)

        return new_df

    @staticmethod
    def num_alpha_splitter(rows):
        new_rows = []
        max_column_num = 0
        for row in rows:
            if row:
                new_row = re.findall(r'[0-9.0-9]+|[a-zA-Z]+', str(row))
                max_column_num = max(max_column_num, len(new_row))
                new_rows.append(new_row)
            else:
                new_rows.append([np.nan])
                max_column_num = max(max_column_num, 1)
        row_count = 0
        while row_count < len(rows):
            if len(new_rows[row_count]) < max_column_num:
                new_rows[row_count] = [np.nan] * max_column_num
            row_count += 1
        new_rows = np.array(new_rows).T
        return new_rows

    @staticmethod
    def is_num_alpha(rows, num_alpha_threshold=0.9):
        match_count = 0
        for row in rows:
            num_alpha_match = re.match(r'[\d]+[A-Za-z]+|[A-Za-z]+[\d]+', str(row))
            if num_alpha_match:
                match_count += 1
        if float(match_count) / len(rows) > num_alpha_threshold:
            return True
        return False

    @staticmethod
    def num_check(rows, num_threshold=0.1):
        num_count = 0
        for row in rows:
            try:
                float(row)
                num_count += 1
                pass
            except Exception:
                pass
        if float(num_count) / len(rows) >= num_threshold:
            return True
        else:
            return False


if __name__ == '__main__':
    file = '/Users/runqishao/Downloads/Archive/LL0_188_eucalyptus/LL0_188_eucalyptus_dataset/tables/learningData.csv'
    df = pd.read_csv(file)

    punc_list = PunctuationParser.detect(df=df.iloc[:50, :])
    na_list = NumAlphaParser.detect(df=df.iloc[:50, :])
    print(punc_list)
    print(na_list)
    df.to_csv('/Users/runqishao/Desktop/aaa.csv')
    punc_result = PunctuationParser.perform(df=df, columns_perform=punc_list)

    punc_result.to_csv('/Users/runqishao/Desktop/bbb.csv')

    na_result = NumAlphaParser.perform(df=punc_result, columns_perform=na_list)

    na_result.to_csv('/Users/runqishao/Desktop/ccc.csv')
