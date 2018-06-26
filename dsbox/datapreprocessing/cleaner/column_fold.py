import pandas as pd
import itertools
from itertools import groupby
from itertools import chain
import dateparser
import json


def process(df):
    """
    this function will actually fold the dataframe if there are any foldable columns discovered
    :param df: input dataframe
    :return: folded df
    """
    columns_names = list(df)
    foldable_columns = columns_to_fold(columns_names)
    """
    lets handle both these cases, first we are going to handle the dates part as more often than not its the sure 
    shot case
    """
    if len(foldable_columns['date_as_column_candidates']) > 0:
        return fold_columns(df, foldable_columns['date_as_column_candidates'], 'date')

    column_name_candidates = foldable_columns['column_name_candidates']
    for key in list(column_name_candidates):
        df = fold_columns(df, column_name_candidates[key], key.strip())

    return df


def process_column_prefix(df):
    """
    Detect foldable columns based on common prefix
    :param df: input dataframe
    :return: folded dataframe if applicable otherwise input datafram
    """
    columns_names = list(df)
    foldable_columns = columns_to_fold(columns_names)

    column_name_candidates = foldable_columns['column_name_candidates']
    for key in list(column_name_candidates):
        df = fold_columns(df, column_name_candidates[key], key.strip())

    return df


def process_column_date(df):
    """
    Detect foldable columns if they are valid dates
    :param df: input data frame
    :return: folded data frame if applicable otherwise input datafram
    """
    columns_names = list(df)
    foldable_columns = columns_to_fold(columns_names)

    if len(foldable_columns['date_as_column_candidates']) > 0:
        return fold_columns(df, foldable_columns['date_as_column_candidates'], 'date')
    return df


def fold_columns(df, columns_to_fold, new_column_name):
    if len(columns_to_fold) == 0:
        # nothing to fold, return the original
        return df

    new_column_name = new_column_name.strip()
    new_rows_list = list()
    orig_columns = df.columns.values

    non_foldable_columns = list(set(orig_columns) - set(columns_to_fold))

    for i in df.index.values:
        row = df.iloc[i]
        for column_to_fold in columns_to_fold:
            d1 = {}
            for nfc in non_foldable_columns:
                d1[nfc] = row[nfc]

            d1[new_column_name] = column_to_fold
            d1['{}_value'.format(new_column_name)] = row[column_to_fold]
            new_rows_list.append(d1)

    new_df = pd.DataFrame(new_rows_list)
    print(new_df)
    return new_df


def columns_to_fold(columns_names):
    result = dict()
    result['column_name_candidates'] = based_on_same_prefix(columns_names)
    result['date_as_column_candidates'] = column_names_dates(columns_names)

    return result


def all_equal(iterable):
    # copied from itertools documentation
    """ Returns True if all the elements are equal to each other"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def powerset(iterable):
    # copied from itertools documentation
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def based_on_same_prefix(column_names, threshold=0.5):
    """
    This function will return a list of prefixes which are candidates for "folding".
    It does the following:
        - separates out column names based on the first character of the column name and created a dictionary with
            first character as the key
        - for each entry in the dictionary, create a powerset of the column names and for each subset do the following
        - find a common prefix as described below

    :param column_names: a list of input column names
    :param threshold: this is the minimum ratio of the prefix found to the max length of the candidate column names
    :return: a list of folding candidate column names
    """
    if len(column_names) <= 1:
        return ""

    column_dict = dict()
    result_columns = dict()
    for i in range(0, len(column_names)):
        s_char = column_names[i][0]
        if s_char not in column_dict:
            column_dict[s_char] = list()
        column_dict[s_char].append(column_names[i])

    for r in list(column_dict):
        if len(column_dict[r]) > 1:
            column_powerset = powerset(column_dict[r])
            for column_subset in column_powerset:
                if len(column_subset) > 0:
                    result = set()
                    prefix = ''
                    common_prefixes = common_prefix(list(column_subset))
                    for c in common_prefixes:
                        # c in this case is a tuple, example: ('t', 't')   get the first value
                        prefix += c[0]

                    if prefix.strip() != '':
                        candidate_column_names = [x for x in column_names if x.startswith(prefix)]
                        # ignore the candidates where prefix is the whole column name
                        if len(candidate_column_names) > 1:
                            max_length = max([len(x) for x in candidate_column_names])
                            if float(len(prefix)) / float(max_length) >= threshold:
                                for c in candidate_column_names:
                                    result.add(c)
                        if len(result) > 0:
                            if prefix not in result_columns:
                                result_columns[prefix] = list(result)
    return result_columns


def column_names_dates(column_names):
    """
    This function will return a list of column names if they are dates
    :param column_names:
    :return: a subset of column_names
    """
    result = list()
    for column_name in column_names:
        # do not want 12 to be parsed as date, minimum length should be 4 (year in YYYY format)
        if len(column_name) >= 4:
            try:
                # for now strict parsing is true, otherwise it'll parse 'year' as valid date.
                # in future, we'll have to specify date formats
                parsed_column_as_date = dateparser.parse(column_name, settings={'STRICT_PARSING': True})
                if parsed_column_as_date:
                    # column_name has been parsed as a valid date, it is a candidate for fold
                    result.append(column_name)
            except:
                # something went wrong, doesn't matter what
                pass
    return result


def common_prefix(its):
    """returns list of tuples of common characters"""
    yield from itertools.takewhile(all_equal, zip(*its))


if __name__ == '__main__':
    input_file = 'data_sample/sample1.csv'
    df = pd.read_csv(input_file)
    process(df)
