#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import numpy as np
import os
import pandas as pd
import random


class DataCleaning:

    def __init__(
        self,
        input_df,
        ignore_list,
        options=0,
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

        self.num_threshold = num_threshold
        self.common_threshold = common_threshold
        self.input_df = input_df
        self.ignore_list = ignore_list

        if options == 0:
            self.return_rlt = self.iterations_punc()
        elif options == 1:
            self.return_rlt = self.iterations_num_alpha()
        elif options == 2:
            self.return_rlt = self.iterations_phone()

    def iterations_phone(self):
        all_columns = self.input_df.columns
        require_checking = \
            list(set(all_columns).difference(set(self.ignore_list)))
        extends = {}
        for one_column in require_checking:
            isphone = self.is_phone(self.input_df[one_column])
            if isphone == True:
                phone_number = \
                    self.phone_parser(self.input_df[one_column])
                extends[one_column + '_0'] = phone_number
        return extends

    def is_phone(self, rows):
        pattern = \
            '^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'
        rows = self.random_50(rows)
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

    def iterations_num_alpha(self):

        all_columns = self.input_df.columns
        require_checking = \
            list(set(all_columns).difference(set(self.ignore_list)))
        extends = {}
        for one_column in require_checking:
            isnum_alpha = self.is_num_alpha(self.input_df[one_column])
            isnum = self.num_check(self.input_df[one_column])
            if isnum == False:
                if isnum_alpha == True:
                    num_alpha = \
                        self.num_alpha_splitter(self.input_df[one_column])
                    count = 0
                    for one in num_alpha:
                        extends[one_column + '_' + str(count)] = one
                        count += 1

        return extends

    def is_num_alpha(self, rows):
        rows = self.random_50(rows)
        match_count = 0
        for row in rows:
            num_alpha_match = re.match(r'[0-9.0-9]+|[a-zA-Z]+',
                    str(row))
            if num_alpha_match != None:
                match_count += 1
        if float(match_count) / len(rows) > 0.5:
            return True
        return False

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

    def iterations_punc(self):
        all_columns = self.input_df.columns
        require_checking = \
            list(set(all_columns).difference(set(self.ignore_list)))
        extends = {}
        for one_column in require_checking:
            isnum = self.num_check(self.input_df[one_column])
            if isnum == False:
                common_list = \
                    self.find_common(self.input_df[one_column])
                if len(common_list) >= 1:
                    count = 0
                    splitted = self.splitter(self.input_df[one_column],
                            common_list)
                    if len(splitted) > 1:
                        for one in splitted:
                            extends[one_column + '_' + str(count)] = one
                            count += 1
        return extends

    def random_50(self, rows):
        rows = list(rows)
        if len(rows) > 50:
            rows = random.sample(rows, 50)
        return rows

    def num_check(self, rows):
        rows = self.random_50(rows)
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
        rows = self.random_50(rows)
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
            new_row = [x for x in re.split(re_list, row) if x]
            max_column_num = max(max_column_num, len(new_row))
            new_rows.append(new_row)

        row_count = 0
        while row_count < len(rows):
            if len(new_rows[row_count]) < max_column_num:
                new_rows[row_count].extend([np.nan] * (max_column_num-len(new_rows[row_count])))
            row_count += 1
        new_rows = np.array(new_rows).T
        return new_rows

    def return_func(self):
        output_df = self.input_df
        if self.return_rlt != None:
            for key in self.return_rlt:
                output_df[key] = self.return_rlt[key]
        return output_df


if __name__ == '__main__':
    file = '/Users/xkgoodbest/Documents/ISI/learningData 2 188.csv'
    df = pd.read_csv(file)
    ignore_list = ['d3mIndex']

    # options: 0 for punctuation split, 1 for num_alpha split, 2 for phone parser

    punc = DataCleaning(df, ignore_list, options=0)
    result = punc.return_func()
