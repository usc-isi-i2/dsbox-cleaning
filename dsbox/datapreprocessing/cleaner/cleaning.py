import time
import logging

from date_featurizer_org import DateFeaturizer
from data_cleaning import DataCleaning

class Cleaner:

    def __init__(self, df, min_threshold=0.95, extractor_settings=None):
        self.df = df
        
        self._date_featurizer = DateFeaturizer(df, min_threshold=min_threshold, extractor_settings=extractor_settings)
        logging.basicConfig(filename='cleaner.log', level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('Started cleaning on dataframe: '+self.df.name)

    def clean_dataframe(self):

        # Create a sample of 50 rows
        sample = self.df.sample(n=50)

        start = time.time()
        # Parse dates and featurize
        try:
            out = self._date_featurizer.featurize_dataframe(sampled_df=sample)
            self.df = out['df']
            self._ignore_date_cols = out['date_columns'] # columns to ignore
        except Exception as e:
            self._ignore_date_cols = []
            logging.error("Date Featurization failed")
            logging.error(str(e))
        
        end = time.time()

        logging.info("Date featurization finished in: "+str(end-start))

        start = time.time()
        try:
            punc_splitter = DataCleaning(self.df, self._ignore_date_cols, options=0)
            self.df = punc_splitter.return_func()
        except Exception as e:
            logging.error("Punctuation splitter failed")
            logging.error(str(e))
        end = time.time()

        logging.info("Punctuation splitter finished in: "+str(end-start))

        start = time.time()
        try:
            num_alpha_splitter = DataCleaning(self.df, self._ignore_date_cols, options=1)
            self.df = num_alpha_splitter.return_func()
        except:
            logging.error("num-apha splitter failed")
            logging.error(str(e))
        end = time.time()

        logging.info("num-apha splitter finished in: "+str(end-start))

        start = time.time()
        try:
            phone_parser = DataCleaning(self.df, self._ignore_date_cols, options=2)
            self.df = phone_parser.return_func()
        except:
            logging.error("phone parser failed")
            logging.error(str(e))
        end = time.time()

        logging.info("phone parser finished in: "+str(end-start))

        return self.df