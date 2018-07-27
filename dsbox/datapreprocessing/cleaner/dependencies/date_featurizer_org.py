import time
from datetime import datetime
import pandas as pd
from warnings import warn
import re
import numpy as np

from dsbox.datapreprocessing.cleaner.dependencies.date_extractor import DateExtractor
from dsbox.datapreprocessing.cleaner.dependencies.helper_funcs import HelperFunction
import d3m.metadata.base as mbase


class DateFeaturizerOrg:

    def __init__(self, dataframe,
                 min_threshold=0.9,
                 create_year=True,
                 create_month=True,
                 create_day=True,
                 create_day_of_week=True,
                 create_epoch=True,
                 drop_original_column=False,
                 extractor_settings=None):

        """
        dataframe: df to featurize
        min_threshold: [0.0 to 1.0] Fraction of values required to be parsed as dates in order to
        featurize the
                        column
        create_<date_resolution>: [Bool] Whether to create the column or not (global)
        drop_original_column: [Bool] Whether to drop the original column after featurizing or not
        extractor_settings: [Dict] Extractor settings for the date parser (see
        dependencies/date_extractor.py)
        """
        self.df = dataframe
        self.min_threshold = min_threshold
        self.create_year = create_year
        self.create_month = create_month
        self.create_day = create_day
        self.create_day_of_week = create_day_of_week
        self.create_epoch = create_epoch
        self.drop_original_column = drop_original_column
        if extractor_settings is not None:
            self.extractor_settings = extractor_settings
        else:
            self.extractor_settings = {}

        self._samples_to_print = []  # Column names to print as a sample
        self.date_extractor = DateExtractor()

        # Original settings saved, do not modify - readonly
        self._crY = create_year
        self._crM = create_month
        self._crD = create_day
        self._crDow = create_day_of_week
        self._crE = create_epoch

        # Month range parser settings
        self._month_range_pattern = r'^\w{3,9}\-\w{3,9}$'
        self._month_range_delim = '-'
        self._month_abbv = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Sep',
                            'Oct', 'Nov', 'Dec']
        self._month_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                            'August', 'September', 'October', 'November', 'December']

    def featurize_date_columns(self, column_indices):
        """
        Featurize date columns in the dataframe (given in column_indices)

        params:
        - column_indices [List]: List of column indices with dates
        """

        for idx in column_indices:
            values = self._parse_column(self.df, idx)
            if values is not None:
                self._featurize_column(values, idx)
            self.create_day = self._crD
            self.create_day_of_week = self._crDow
            self.create_epoch = self._crE
            self.create_month = self._crM
            self.create_year = self._crY

        return {
            'df': self.df,
            'date_columns': self._samples_to_print
        }

    def detect_date_columns(self, sampled_df, except_list=list()):
        """
        Detects date columns in the sampled_df and returns a list of column indices which have dates

        params:
        - sampled_df [DataFrame]: a sample of rows from the original dataframe for detecting dates
        - except_list [List]: list of column indices to be ignored
        """
        positive_semantic_types = set(["https://metadata.datadrivendiscovery.org/types/Time",
                                       "http://schema.org/Text"])
        cols_to_detect = HelperFunction.cols_to_clean(sampled_df, positive_semantic_types)

        date_cols = []
        for idx in cols_to_detect:
            if idx not in except_list:
                if self._parse_column(sampled_df, idx) is not None:
                    date_cols.append(idx)
        return date_cols

    def _featurize_column(self, values, idx):
        """
        Featurize a column that has been parsed
        """
        years = []
        days = []
        months = []
        dows = []
        epochs = []

        column_label = self.df.columns[idx]

        for x in values:
            if self.create_year:
                years.append(x.year if x is not None else None)
            if self.create_month:
                months.append(x.month if x is not None else None)
            if self.create_day:
                days.append(x.day if x is not None else None)
            if self.create_day_of_week:
                dows.append(x.isoweekday() if x is not None else None)
            if self.create_epoch:
                if x is not None:
                    try:
                        epoch = time.mktime(x.timetuple())
                    except OverflowError as e:
                        epoch = None
                        print(e)
                else:
                    epoch = None
                epochs.append(epoch)
        if self.create_year:
            self.df[column_label + "_year"] = years
            self.update_types(column_label + "_year")
            self._samples_to_print.append(self.df.columns.get_loc(column_label + "_year") - 1)
        if self.create_month:
            self.df[column_label + "_month"] = months
            self.update_types(column_label + "_month")
            self._samples_to_print.append(self.df.columns.get_loc(column_label + "_month") - 1)
        if self.create_day:
            self.df[column_label + "_day"] = days
            self.update_types(column_label + "_day")
            self._samples_to_print.append(self.df.columns.get_loc(column_label + "_day") - 1)
        if self.create_day_of_week:
            self.df[column_label + "_day_of_week"] = dows
            self.update_types(column_label + "_day_of_week")
            self._samples_to_print.append(
                self.df.columns.get_loc(column_label + "_day_of_week") - 1)
        if self.create_epoch:
            self.df[column_label + "_epochs"] = epochs
            self.update_types(column_label + "_epochs")
            self._samples_to_print.append(self.df.columns.get_loc(column_label + "_epochs") - 1)

    def _parse_month_range(self, df, idx):
        pattern = re.compile(self._month_range_pattern)

        parsed_values = []

        for item in df.iloc[:, idx]:

            # remove whitespace
            item = str(item).strip()

            if pattern.match(item):
                item = item.split(self._month_range_delim)

                if item[0] in self._month_abbv and item[1] in self._month_abbv:
                    parsed_values.append(item)
                elif item[0] in self._month_full and item[1] in self._month_full:
                    parsed_values.append(item)
                else:
                    parsed_values.append(None)
            else:
                parsed_values.append(None)

        frac_parsed = 1 - ((parsed_values.count(None) - df.iloc[:, idx].isnull().sum())
                           / len(parsed_values))

        if frac_parsed >= self.min_threshold:
            return parsed_values

        return None

    def _parse_month(self, df, idx):

        parsed_values = []

        for item in df.iloc[:, idx]:

            # remove whitespace
            item = str(item).strip()

            try:
                item = datetime.strptime(item, "%b")
            except ValueError:
                parsed_values.append(None)
            else:
                parsed_values.append(item)

        frac_parsed = 1 - ((parsed_values.count(None) - df.iloc[:, idx].isnull().sum())
                           / len(parsed_values))

        if frac_parsed >= self.min_threshold:
            return parsed_values
        else:
            parsed_values = []
            for item in df.iloc[:, idx]:
                # remove whitespace
                item = str(item).strip()

                try:
                    item = datetime.strptime(item, "%B")
                except ValueError:
                    parsed_values.append(None)
                else:
                    parsed_values.append(item)

            frac_parsed = 1 - ((parsed_values.count(None) - df.iloc[:, idx].isnull().sum())
                               / len(parsed_values))

            if frac_parsed >= self.min_threshold:
                return parsed_values
            else:
                return None

    def _parse_weekday(self, df, idx):

        parsed_values = []

        for item in df.iloc[:, idx]:
            item = str(item).strip()
            try:
                item = datetime.strptime(item, "%A")
            except ValueError:
                try:
                    item = datetime.strptime(item, "%a")
                except ValueError:
                    parsed_values.append(None)
                else:
                    parsed_values.append(item)
            else:
                parsed_values.append(item)

        frac_parsed = 1 - ((parsed_values.count(None) - df.iloc[:, idx].isnull().sum())
                           / len(parsed_values))

        if frac_parsed >= self.min_threshold:
            return parsed_values
        else:
            # print(column_label," does not qualify")
            # print(frac_parsed)
            return None

    def _parse_column(self, df, idx):
        """
        Parse column and detect dates
        """

        # Do not parse float values
        if df.iloc[:, idx].dtype == float:
            return None

        parsed_values = []
        multiple_values = False

        custom_settings = dict(self.extractor_settings)
        custom_settings['additional_formats'] = map(lambda s: r'[^\d.]' + s + r'[^\d.]',
                                                    [r'D-%d/%m/%y', r'%m00%y', r"%Y%m%d",
                                                     r"%a %B %d %H:%M:%S EDT %Y",
                                                     r"%a %B %d %H:%M:%S %Z %Y"])
        # lambda s: r'[^\d.]'+s+r'[^\d.]'
        custom_settings['use_default_formats'] = False

        month_parsed_values = self._parse_month(df, idx)

        if self._parse_month_range(df, idx) is not None:
            # Do not parse month ranges
            warn("Month range ignored")
            return None

        if month_parsed_values is not None:
            # change featurization settings
            self.create_day = False
            self.create_day_of_week = False
            self.create_epoch = False
            self.create_month = True
            self.create_year = False
            return month_parsed_values

        if self._parse_weekday(df, idx) is not None:
            warn("Weekday ignored")
            return None

        for item in df.iloc[:, idx]:

            extracted = self.date_extractor.extract(str(item), **custom_settings)

            if len(extracted) == 0:
                extracted = self.date_extractor.extract(str(item), **self.extractor_settings)

            if len(extracted) > 0:
                if len(extracted) > 1:
                    multiple_values = True
                parsed_values.append(extracted[0])
            else:
                parsed_values.append(None)
        if multiple_values:
            warn("Warning: multiple dates detected in column: " + str(idx))

        frac_parsed = 1 - ((parsed_values.count(None) - df.iloc[:, idx].isnull().sum())
                           / len(parsed_values))

        if frac_parsed >= self.min_threshold:
            return parsed_values
        else:
            return None

    def print_sample(self, input_filename):
        # Put random 20 rows of the dataset with the parsed dates into a sample csv
        if self.df.shape[0] > 20:
            N = 20
        else:
            N = self.df.shape[0]
        self.df.iloc[:, self._samples_to_print] \
            .sample(n=N) \
            .to_csv(input_filename + "_sample.csv")

    def featurize_dataframe(self):
        """
        Detect and featurize together
        """
        # Create sample of 50 rows
        sample = self.df.sample(n=50)

        # Detect date columns
        date_cols = self.detect_date_columns(sample)

        # Featurize date columns
        return self.featurize_date_columns(date_cols)

    def update_types(self, col_name):
        old_metadata = dict(
            self.df.metadata.query((mbase.ALL_ELEMENTS, self.df.columns.get_loc(col_name))))

        numerics = pd.to_numeric(self.df[col_name], errors='coerce')
        length = numerics.shape[0]
        nans = numerics.isnull().sum()

        if nans / length > 0.9:
            if HelperFunction.is_categorical(self.df[col_name]):
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

        old_metadata['semantic_types'] += \
            ("https://metadata.datadrivendiscovery.org/types/Attribute",)

        self.df.metadata = self.df.metadata.update(
            (mbase.ALL_ELEMENTS, self.df.columns.get_loc(col_name)), old_metadata)
