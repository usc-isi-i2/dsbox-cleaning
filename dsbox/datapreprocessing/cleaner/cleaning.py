import time
import logging

from date_featurizer_org import DateFeaturizer
from data_cleaning import NumAlphaSplitter,PhoneParser,PunctuationSplitter

class Cleaner:

    def __init__(self, df, min_threshold=0.95, extractor_settings=None):
        self.df = df
        
        self._date_featurizer = DateFeaturizer(df, min_threshold=min_threshold, extractor_settings=extractor_settings)
        logging.basicConfig(filename='cleaner.log', level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('Started cleaning on dataframe: '+self.df.name)

    def clean_dataframe(self):

        start = time.time()
        # Parse dates and featurize
        try:
            out = self._date_featurizer.featurize_dataframe()
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
            punc_splitter_detection = PunctuationSplitter(self.df, doing_list=self._ignore_date_cols,
                    options=0, num_threshold=0.1, common_threshold=0.9)
            doing_list = punc_splitter_detection.return_results()

            punc_splitter = PunctuationSplitter(self.df, doing_list,
                    options=1, num_threshold=0.1, common_threshold=0.9)
            self.df = punc_splitter.return_results()

        except Exception as e:
            logging.error("Punctuation splitter failed")
            logging.error(str(e))
        end = time.time()

        logging.info("Punctuation splitter finished in: "+str(end-start))

        start = time.time()
        try:
            na_splitter_detection = NumAlphaSplitter(self.df, doing_list=self._ignore_date_cols, options=0,
                                   num_threshold=0.1,
                                   num_alpha_threshold=0.8)
            doing_list = na_splitter_detection.return_results()

            na_splitter = NumAlphaSplitter(self.df, doing_list=doing_list, options=1,
                                   num_threshold=0.1,
                                   num_alpha_threshold=0.8)
            self.df = na_splitter.return_results()

        except:
            logging.error("num-apha splitter failed")
            logging.error(str(e))
        end = time.time()

        logging.info("num-apha splitter finished in: "+str(end-start))

        start = time.time()
        try:
            phone_parser_detection = PhoneParser(self.df, doing_list=self._ignore_date_cols, options=0)
            doing_list = phone_parser_detection.return_results()
            phone_parser = PhoneParser(self.df, doing_list=doing_list, options=1)
            self.df = phone_parser.return_results()

        except:
            logging.error("phone parser failed")
            logging.error(str(e))
        end = time.time()

        logging.info("phone parser finished in: "+str(end-start))

        return self.df