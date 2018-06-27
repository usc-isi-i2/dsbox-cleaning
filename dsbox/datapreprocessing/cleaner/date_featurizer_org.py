import time
from datetime import datetime
import pandas as pd
from warnings import warn
import re

from dependencies.date_extractor import DateExtractor


class DateFeaturizer:

	def __init__(self, dataframe, 
		min_threshold=0.9,
		create_year=True,
		create_month=True,
		create_day=True,
		create_day_of_week=True,
		create_epoch=True,
		extractor_settings=None):

		"""
		dataframe: df to featurize
		min_threshold: [0.0 to 1.0] Fraction of values required to be parsed as dates in order to featurize the
						column
		create_<date_resolution>: [Bool] Whether to create the column or not (global)
		extractor_settings: [Dict] Extractor settings for the date parser (see dependencies/date_extractor.py)
		"""

		self.df = pd.DataFrame(dataframe)
		self.min_threshold = min_threshold
		self.create_year = create_year
		self.create_month=create_month
		self.create_day=create_day
		self.create_day_of_week=create_day_of_week
		self.create_epoch=create_epoch
		if extractor_settings is not None:
			self.extractor_settings=extractor_settings
		else:
			self.extractor_settings={}

		self._samples_to_print = []
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
		self._month_abbv = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Sep', 'Oct', 'Nov', 'Dec']
		self._month_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

	def featurize_dataframe(self, sampled_df=None):
		"""
		Featurize all date values in the dataframe 

		sampled_df: [dataframe] Df that contains a sample of rows from original df
		"""
		parsed_columns = []
		
		if sampled_df is not None:
			parsed_columns = self.sample_dataframe(sampled_df)
		else:
			parsed_columns = self.df.columns.values

		for column_label in parsed_columns:
			values = self._parse_column(self.df, column_label)
			if values is not None:
				self._featurize_column(values, column_label)
			self.create_day = self._crD
			self.create_day_of_week = self._crDow
			self.create_epoch = self._crE
			self.create_month = self._crM
			self.create_year = self._crY
		
		self.df = self.df.drop(parsed_columns, axis=1)
		return {
			'df':self.df,
			'date_columns':self._samples_to_print
		}

	def _featurize_column(self, values, column_label):
		"""
		Featurize a column that has been parsed 
		"""
		years = []
		days = []
		months = []
		dows = []
		epochs = []
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
			self.df[column_label+"_year"] = years
			self._samples_to_print.append(column_label+"_year")
		if self.create_month:
			self.df[column_label+"_month"] = months
			self._samples_to_print.append(column_label+"_month")
		if self.create_day:
			self.df[column_label+"_day"] = days
			self._samples_to_print.append(column_label+"_day")
		if self.create_day_of_week:
			self.df[column_label+"_day_of_week"] = dows
			self._samples_to_print.append(column_label+"_day_of_week")
		if self.create_epoch:
			self.df[column_label+"_epochs"] = epochs
			self._samples_to_print.append(column_label+"_epochs")

	def _parse_month_range(self, df, column_label):
		pattern = re.compile(self._month_range_pattern)
		
		parsed_values = []

		for item in df[column_label]:

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
		
		frac_parsed = 1 - ((parsed_values.count(None) - df[column_label].isnull().sum())/len(parsed_values))

		if frac_parsed >= self.min_threshold:
			return parsed_values
		
		return None

	
	def _parse_month(self, df, column_label):

		parsed_values = []

		for item in df[column_label]:

			# remove whitespace
			item = str(item).strip()

			try:
				item = datetime.strptime(item, "%b")
			except ValueError:
				parsed_values.append(None)
			else:
				parsed_values.append(item)
		
		frac_parsed = 1 - ((parsed_values.count(None) - df[column_label].isnull().sum())/len(parsed_values))

		if frac_parsed >= self.min_threshold:
			return parsed_values
		else:
			parsed_values = []
			for item in df[column_label]:
				# remove whitespace
				item = str(item).strip()

				try:
					item = datetime.strptime(item, "%B")
				except ValueError:
					parsed_values.append(None)
				else:
					parsed_values.append(item)
			
			frac_parsed = 1 - ((parsed_values.count(None) - df[column_label].isnull().sum())/len(parsed_values))

			if frac_parsed >= self.min_threshold:
				return parsed_values
			else:
				return None

	def _parse_weekday(self, df, column_label):
	
		parsed_values = []

		for item in df[column_label]:
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
		
		frac_parsed = 1 - ((parsed_values.count(None) - df[column_label].isnull().sum())/len(parsed_values))

		if frac_parsed >= self.min_threshold:
			return parsed_values
		else:
			# print(column_label," does not qualify")
			# print(frac_parsed) 
			return None

	def _parse_column(self, df, column_label):
		"""
		Parse column and detect dates
		"""

		# Do not parse float values
		if df[column_label].dtype == float:
			return None

		parsed_values = []
		multiple_values = False

		custom_settings = dict(self.extractor_settings)
		custom_settings['additional_formats']=['D-%d/%m/%y', '%m00%y', "%Y%m%d", "%a %B %d %H:%M:%S EDT %Y", "%a %B %d %H:%M:%S %Z %Y"]
		custom_settings['use_default_formats']=False

		month_parsed_values = self._parse_month(df, column_label)

		if self._parse_month_range(df, column_label) is not None:
			# Do not parse month ranges
			warn("Month range ignored")
			print(column_label)
			return None

		if month_parsed_values is not None:
			# change featurization settings
			self.create_day = False
			self.create_day_of_week = False
			self.create_epoch = False
			self.create_month = True
			self.create_year = False
			return month_parsed_values
		
		if self._parse_weekday(df, column_label) is not None:
			warn("Weekday ignored")
			return None

		for item in df[column_label]:

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
			warn("Warning: multiple dates detected in column: "+column_label)

		frac_parsed = 1 - ((parsed_values.count(None) - df[column_label].isnull().sum())/len(parsed_values))

		if frac_parsed >= self.min_threshold:
			return parsed_values
		else:
			# print(column_label," does not qualify")
			# print(frac_parsed) 
			return None

	def print_sample(self, input_filename):
		# Put random 20 rows of the dataset with the parsed dates into a sample csv
		if self.df.shape[0] > 20:
			N = 20
		else:
			N = self.df.shape[0]
		self.df[self._samples_to_print] \
			.sample(n=N) \
			.to_csv(input_filename+"_sample.csv")

	def sample_dataframe(self, sampled_df):
		date_cols = []
		for column_label in self.df.columns.values:
			if self._parse_column(sampled_df,column_label) is not None:
				date_cols.append(column_label)
		return date_cols
