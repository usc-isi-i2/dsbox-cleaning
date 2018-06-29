#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import numpy as np
import os
import pandas as pd


class PhoneParser:

    def __init__(
        self,
        df,
        doing_list,
        options,
        ):
        self.df = df
        self.doing_list = doing_list
        if options == 0:
            self.phone_list = self.detection()
        elif options == 1:
            self.phones = self.performing()
            self.phone_list = self.df
            for key in self.phones:
                self.phone_list[key] = self.phones[key]

    def detection(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.doing_list)))
        extends = []
        for one_column in require_checking:
            isphone = self.is_phone(self.df.iloc[:, one_column])
            if isphone == True:
                extends.append(one_column)
        return extends

    def performing(self):
        extends = {}
        for one_column in self.doing_list:
            restult = self.phone_parser(self.df.iloc[:, one_column])
            extends[self.df.columns[one_column] + '_phone'] = restult
        return extends

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

    def return_results(self):
        return self.phone_list


class PunctuationSplitter:

    def __init__(
        self,
        df,
        doing_list,
        options,
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
        self.doing_list = doing_list
        self.num_threshold = num_threshold
        self.common_threshold = common_threshold
        if options == 0:
            self.punc_columns = self.detection()
        elif options == 1:
            self.splitted = self.performing()
            self.punc_columns = self.df
            for key in self.splitted:
                self.punc_columns[key] = self.splitted[key]

    def detection(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.doing_list)))
        extends = []
        for one_column in require_checking:
            isnumber = self.num_check(self.df.iloc[:, one_column])
            if isnumber == False:
                common_list = self.find_common(self.df.iloc[:,
                        one_column])
                if len(common_list) > 0:
                    extends.append(one_column)
        return extends

    def performing(self):
        extends = {}
        for one_column in self.doing_list:
            common_list = self.find_common(self.df.iloc[:, one_column])
            restult = self.splitter(self.df.iloc[:, one_column],
                                    common_list)
            count = 0
            for one in restult:
                extends[self.df.columns[one_column] + '_punc_'
                        + str(count)] = one
                count += 1
        return extends

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

    def return_results(self):
        return self.punc_columns


class NumAlphaSplitter:

    def __init__(
        self,
        df,
        doing_list,
        options,
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
        self.doing_list = doing_list
        self.num_threshold = num_threshold
        self.num_alpha_threshold = num_alpha_threshold
        if options == 0:
            self.na_columns = self.detection()
        elif options == 1:
            self.splitted = self.performing()
            self.na_columns = self.df
            for key in self.splitted:
                self.na_columns[key] = self.splitted[key]

    def detection(self):
        all_indices = range(0, self.df.shape[1])
        require_checking = \
            list(set(all_indices).difference(set(self.doing_list)))
        extends = []
        for one_column in require_checking:
            isnumber = self.num_check(self.df.iloc[:, one_column])
            if isnumber == False:
                isnum_alpha = self.is_num_alpha(self.df.iloc[:,
                        one_column])
                if isnum_alpha == True:
                    extends.append(one_column)
        return extends

    def performing(self):
        extends = {}
        for one_column in self.doing_list:
            restult = self.num_alpha_splitter(self.df.iloc[:,
                    one_column])
            count = 0
            for one in restult:
                extends[self.df.columns[one_column] + '_na_'
                        + str(count)] = one
                count += 1
        return extends

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

    def return_results(self):
        return self.na_columns

if __name__ == '__main__':
    file = '/Users/xkgoodbest/Documents/ISI/my_data_cleaning/learningData 2 188.csv'
    df = pd.read_csv(file)

    '''
    options: 0 for detection, doing_list works as ignore_list. output is a list of indices
    options: 1 for performing, doing_list works as perform_list. output is a new dataframe
    '''
    phone_parser = PhoneParser(df, doing_list=[6], options=1)
    phone_list = phone_parser.return_results()



    '''
    options: 0 for detection, doing_list works as ignore_list. output is a list of indices
    options: 1 for performing, doing_list works as perform_list. output is a new dataframe
    '''
    punc_splitter = PunctuationSplitter(df, doing_list=[3, 4, 5],
            options=1, num_threshold=0.1, common_threshold=0.9)
    punc_result = punc_splitter.return_results()


    '''
    options: 0 for detection, doing_list works as ignore_list. output is a list of indices
    options: 1 for performing, doing_list works as perform_list. output is a new dataframe
    '''
    na_splitter = NumAlphaSplitter(df, doing_list=[1, 4, 7], options=1,
                                   num_threshold=0.1,
                                   num_alpha_threshold=0.8)
    na_result = na_splitter.return_results()



