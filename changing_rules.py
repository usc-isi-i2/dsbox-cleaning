
changed_names_cleaning = {
    "dsbox.CleaningFeaturizer": "data_cleaning.CleaningFeaturizer.DSBOX",
    "dsbox.Encoder": "data_preprocessing.Encoder.DSBOX",
    "dsbox.UnaryEncoder": "data_preprocessing.UnaryEncoder.DSBOX",
    "dsbox.GreedyImputation": "data_preprocessing.GreedyImputation.DSBOX",
    "dsbox.IterativeRegressionImputation": "data_preprocessing.IterativeRegressionImputation.DSBOX",
    "dsbox.MeanImputation": "data_preprocessing.MeanImputation.DSBOX",
    "dsbox.IQRScaler": "data_preprocessing.Encoder.DSBOX.IQRScaler.DSBOX",
    "dsbox.Labler": "data_preprocessing.Labler.DSBOX",
    "dsbox.Denormalize": "data_preprocessing.Denormalize.DSBOX", # DATA_TRANSFORMATION
    "dsbox.Profiler": "data_preprocessing.Profiler.DSBOX",
    "dsbox.FoldColumns": "data_preprocessing.FoldColumns.DSBOX",
    "dsbox.Voter": "data_preprocessing.Voter.DSBOX", # ensemble
    "dsbox.VerticalConcat": "data_preprocessing.VerticalConcat.DSBOX",
    "dsbox.EnsembleVoting": "data_preprocessing.EnsembleVoting.DSBOX",
    "dsbox.Unfold": "data_preprocessing.Unfold.DSBOX",
    "dsbox.HorizontalConcat": "data_preprocessing.HorizontalConcat.DSBOX",
    "datamart.Augmentation": "data_augmentation.Augmentation.DSBOX",
    "datamart.QueryDataframe": "data_augmentation.QueryDataframe.DSBOX"
}

changed_names_featurizer = {
    "dsbox.DoNothing": "data_preprocessing.DoNothing.DSBOX",
    "dsbox.MultiTableFeaturization": "data_preprocessing.MultiTableFeaturization.DSBOX",
    "dsbox.DataFrameToTensor": "data_preprocessing.DataFrameToTensor.DSBOX",
    "dsbox.Vgg16ImageFeature": "data_preprocessing.Vgg16ImageFeature.DSBOX",
    "dsbox.ResNet50ImageFeature": "data_preprocessing.ResNet50ImageFeature.DSBOX",
    "dsbox.TimeseriesToList": "data_preprocessing.TimeseriesToList.DSBOX",
    "dsbox.RandomProjectionTimeSeriesFeaturization": "data_preprocessing.RandomProjectionTimeSeriesFeaturization.DSBOX",
    "dsbox.GroupUpByTimeSeries": "data_preprocessing.GroupUpByTimeSeries.DSBOX",
    "dsbox.AutoArima": "time_series_forecasting.AutoArima.DSBOX",
    "dsbox.RNNTimeSeries": "time_series_forecasting.RNNTimeSeries.DSBOX"
}


d3m_path = "d3m.primitives"


# d3m_path + primitive_family(lower case) + name + kind
# kind: dsbox 

