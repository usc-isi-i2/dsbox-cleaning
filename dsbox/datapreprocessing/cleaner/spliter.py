#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import numpy as np
import os
import pandas as pd
import d3m.metadata.base as mbase


def update_type(extends, df_origin):
    new_df = df_origin
    indices = list()
    for key in extends:
        new_df[key] = extends[key]
        indices.append(new_df.columns.get_loc(key))
    for idx in indices:
        old_metadata = dict(new_df.metadata.query((mbase.ALL_ELEMENTS, idx)))

        numerics = pd.to_numeric(new_df.iloc[:, idx], errors='coerce')
        length = numerics.shape[0]
        nans = numerics.isnull().sum()

        if nans / length > 0.9:
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

    def __init__(
            self,
            df,
            columns_ignore=list()
    ):
        self.df = df
        self.columns_ignore = columns_ignore

    def detect(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.columns_ignore)))
        extends = []
        for one_column in require_checking:
            isphone = self.is_phone(self.df.iloc[:, one_column])
            if isphone:
                extends.append(one_column)
        return extends

    def perform(self, columns_perform):
        extends = {}
        for one_column in columns_perform:
            result = self.phone_parser(self.df.iloc[:, one_column])
            extends[self.df.columns[one_column] + '_phone'] = result

        new_df = update_type(extends, self.df)

        return new_df

    def is_phone(self, rows):
        pattern = \
            '^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'
        match_count = 0
        for row in rows:
            phone_match = re.match(pattern, str(row))
            if phone_match != None:
                match_count += 1
        if float(match_count) / len(rows) > 0.5:
            return True
        return False

    def phone_parser(self, rows):
        pattern = \
            '^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'
        new_rows = []
        for row in rows:
            phone_match = re.match(pattern, str(row))
            number = ''
            group_id = 1
            if phone_match != None:
                number = ''
                while group_id < 5:
                    if phone_match.group(group_id) != None:
                        number += phone_match.group(group_id) + '-'
                    group_id += 1
            number = number.strip('-')
            new_rows.append(number)
        return new_rows


class PunctuationParser:

    def __init__(
            self,
            df,
            columns_ignore=list(),
            num_threshold=0.1,
            common_threshold=0.9,
    ):
        '''
        The parameter common_threshold means punctuation density in a column, determined
        by the number of row that contains a specific punctuation to the number of rows.
        num-threshold is number density of a column, determined by the number of float or
        integer to the number of rows.A large common_threshold means most of rows contains
        the specific punctuation. A large num_threshold means number dominates that column.
        '''

        self.df = df
        self.columns_ignore = columns_ignore
        self.num_threshold = num_threshold
        self.common_threshold = common_threshold

    def detect(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.columns_ignore)))
        extends = []
        split_times = []
        columns_to_split = []
        for one_column in require_checking:
            isnumber = self.num_check(self.df.iloc[:, one_column])
            if not isnumber:
                common_list = self.find_common(self.df.iloc[:,
                                               one_column])
                if len(common_list) > 0:
                    result = self.splitter(self.df.iloc[:, one_column],
                                           common_list)

                    columns_to_split.append(one_column)
                    split_times.append(len(result))
        extends.append(columns_to_split)
        extends.append(split_times)
        return extends

    def perform(self, columns_perform):
        extends = {}
        for i, one_column in enumerate(columns_perform[0]):
            common_list = self.find_common(self.df.iloc[:, one_column])
            result = self.splitter(self.df.iloc[:, one_column],
                                   common_list)
            if len(result) > columns_perform[1][i]:
                result.pop()
            elif len(result) < columns_perform[1][i]:
                for j in range(columns_perform[1][i] - len(result)):
                    result.append([np.nan] * len(result[0]))
            count = 0
            for one in result:
                extends[self.df.columns[one_column] + '_punc_'
                        + str(count)] = one
                count += 1

        new_df = update_type(extends[0], self.df)

        return new_df

    def splitter(self, rows, common_list):
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
                new_rows[row_count].extend([np.nan] * (max_column_num
                                                       - len(new_rows[row_count])))
            row_count += 1
        new_rows = np.array(new_rows).T
        return new_rows

    def num_check(self, rows):
        num_count = 0
        for row in rows:
            try:
                float(row)
                num_count += 1
                pass
            except Exception:
                pass
        if float(num_count) / len(rows) >= self.num_threshold:
            return True
        else:
            return False

    def find_common(self, rows):
        common_list = []
        appear_dict = {}
        for row in rows:
            for ch in str(row):
                if (ch.isdigit() or ch.isalpha() or ch == '.') == False:
                    if ch in appear_dict:
                        appear_dict[ch] += 1
                    else:
                        appear_dict[ch] = 1
        for key in appear_dict:
            if float(appear_dict[key]) / len(rows) \
                    >= self.common_threshold:
                common_list.append(key)
        return common_list


class NumAlphaParser:

    def __init__(
            self,
            df,
            columns_ignore=list(),
            num_threshold=0.1,
            num_alpha_threshold=0.8,
    ):
        '''
        The parameter common_threshold means punctuation density in a column, determined
        by the number of row that contains a specific punctuation to the number of rows.
        num-num_alpha_threshold is number_alpha density of a column, determined by the
        number of pattern num_alpha or alpha_num to the number of rows.A large common_threshold
        means most of rows contains the specific punctuation. A large num_threshold means
        number dominates that column.
        '''

        self.df = df
        self.columns_ignore = columns_ignore
        self.num_threshold = num_threshold
        self.num_alpha_threshold = num_alpha_threshold

    def detect(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.columns_ignore)))
        extends = []
        split_times = []
        columns_to_split = []
        for one_column in require_checking:
            isnumber = self.num_check(self.df.iloc[:, one_column])
            if not isnumber:
                isnum_alpha = self.is_num_alpha(self.df.iloc[:,
                                                one_column])
                if isnum_alpha:
                    result = self.num_alpha_splitter(self.df.iloc[:, one_column])
                    columns_to_split.append(one_column)
                    split_times.append(len(result))
        extends.append(columns_to_split)
        extends.append(split_times)
        return extends

    def perform(self, columns_perform):
        extends = {}
        for i, one_column in enumerate(columns_perform[0]):
            result = self.num_alpha_splitter(self.df.iloc[:,
                                             one_column])
            if len(result) > columns_perform[1][i]:
                result.pop()
            elif len(result) < columns_perform[1][i]:
                for j in range(columns_perform[1][i] - len(result)):
                    result.append([np.nan] * len(result[0]))
            count = 0
            for one in result:
                extends[self.df.columns[one_column] + '_na_'
                        + str(count)] = one
                count += 1

        new_df = update_type(extends[0], self.df)

        return new_df

    def num_alpha_splitter(self, rows):
        new_rows = []
        max_column_num = 0
        for row in rows:
            if row != None:
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

    def is_num_alpha(self, rows):
        match_count = 0
        for row in rows:
            num_alpha_match = re.match(r'[\d]+[A-Za-z]+|[A-Za-z]+[\d]+'
                                       , str(row))
            if num_alpha_match != None:
                match_count += 1
        if float(match_count) / len(rows) > self.num_alpha_threshold:
            return True
        return False

    def num_check(self, rows):
        num_count = 0
        for row in rows:
            try:
                float(row)
                num_count += 1
                pass
            except Exception:
                pass
        if float(num_count) / len(rows) >= self.num_threshold:
            return True
        else:
            return False


if __name__ == '__main__':
    file = '/Users/xkgoodbest/Documents/ISI/LL0/LL0_188_eucalyptus/LL0_188_eucalyptus_dataset/tables/learningData.csv'
    df = pd.read_csv(file)

    print(df[:5])
    phone_parser = PhoneParser(df, columns_ignore=[6])
    phone_list = phone_parser.detect()
    phone_result = phone_parser.perform(phone_list)
    print(phone_result[:5])
    phone_result.to_csv('phone.csv')

    punc_splitter = PunctuationParser(df, columns_ignore=[], num_threshold=0.1, common_threshold=0.9)
    punc_list = punc_splitter.detect()
    punc_result = punc_splitter.perform(punc_list)

    print(punc_result[:5])
    punc_result.to_csv('punc.csv')

    na_splitter = NumAlphaParser(df, columns_ignore=[], num_threshold=0.1, num_alpha_threshold=0.8)
    na_list = na_splitter.detect()
    na_result = na_splitter.perform(na_list)

    print(na_result[:5])
    na_result.to_csv('na.csv')