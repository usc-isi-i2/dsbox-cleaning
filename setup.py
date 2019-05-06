from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # This is used so that it will pass d3m metadata submission process. The dsbox-featurizer package depends on
        import subprocess
        result = subprocess.check_output(['pip', 'list'])
        lines = str(result).split('\\n')
        for line in lines[2:]:
            part = line.split()
            if 'dsbox-featurizer' in part[0]:
                print(line)
                if '0' == part[1].split('.')[0]:
                    subprocess.call(['pip', 'uninstall', '-y', 'dsbox-featurizer'])
        install.run(self)


setup(name='dsbox-datacleaning',
      version='1.4.4',
      description='DSBox data processing tools for cleaning data',
      author='USC ISI',
      url='https://github.com/usc-isi-i2/dsbox-cleaning.git',
      maintainer_email='kyao@isi.edu',
      maintainer='Ke-Thia Yao',
      license='MIT',
      packages=['dsbox', 'dsbox.datapreprocessing', 'dsbox.datapreprocessing.cleaner', 'dsbox.datapostprocessing'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'scipy>=0.19.0', 'numpy>=1.11.1', 'pandas>=0.20.1', 'langdetect>=1.0.7',
          'scikit-learn>=0.18.0', 'python-dateutil>=2.5.2', 'six>=1.10.0',
          'fancyimpute==0.3.1', 'stopit'
      ],
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'data_cleaning.cleaning_featurizer.DSBOX = dsbox.datapreprocessing.cleaner:CleaningFeaturizer',
              'data_preprocessing.encoder.DSBOX = dsbox.datapreprocessing.cleaner:Encoder',
              'data_preprocessing.unary_encoder.DSBOX = dsbox.datapreprocessing.cleaner:UnaryEncoder',
              'data_preprocessing.greedy_imputation.DSBOX = dsbox.datapreprocessing.cleaner:GreedyImputation',
              'data_preprocessing.iterative_regression_imputation.DSBOX = dsbox.datapreprocessing.cleaner:IterativeRegressionImputation',
              'data_preprocessing.mean_imputation.DSBOX = dsbox.datapreprocessing.cleaner:MeanImputation',
              'normalization.iqr_scaler.DSBOX = dsbox.datapreprocessing.cleaner:IQRScaler',
              'data_cleaning.labeler.DSBOX = dsbox.datapreprocessing.cleaner:Labler',
              'normalization.denormalize.DSBOX = dsbox.datapreprocessing.cleaner:Denormalize',
              'schema_discovery.profiler.DSBOX = dsbox.datapreprocessing.cleaner:Profiler',
              'data_cleaning.column_fold.DSBOX = dsbox.datapreprocessing.cleaner:FoldColumns',
              'data_preprocessing.vertical_concat.DSBOX = dsbox.datapostprocessing:VerticalConcat',
              'data_preprocessing.ensemble_voting.DSBOX = dsbox.datapostprocessing:EnsembleVoting',
              'data_preprocessing.unfold.DSBOX = dsbox.datapostprocessing:Unfold',
              'data_preprocessing.splitter.DSBOX = dsbox.datapreprocessing.cleaner:Splitter',
              'data_preprocessing.horizontal_concat.DSBOX = dsbox.datapostprocessing:HorizontalConcat',
              'data_transformation.to_numeric.DSBOX = dsbox.datapreprocessing.cleaner:ToNumeric'
          ],
      },
      cmdclass={
          'install': PostInstallCommand
      })
