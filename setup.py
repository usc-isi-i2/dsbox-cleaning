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
          'stopit'
      ],
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'data_cleaning.CleaningFeaturizer.DSBOX = dsbox.datapreprocessing.cleaner:CleaningFeaturizer',
              'data_preprocessing.Encoder.DSBOX = dsbox.datapreprocessing.cleaner:Encoder',
              'data_preprocessing.UnaryEncoder.DSBOX = dsbox.datapreprocessing.cleaner:UnaryEncoder',
              'data_preprocessing.GreedyImputation.DSBOX = dsbox.datapreprocessing.cleaner:GreedyImputation',
              'data_preprocessing.IterativeRegressionImputation.DSBOX = dsbox.datapreprocessing.cleaner:IterativeRegressionImputation',
              #              'dsbox.MiceImputation = dsbox.datapreprocessing.cleaner:MICE',
              #              'dsbox.KnnImputation = dsbox.datapreprocessing.cleaner:KNNImputation',
              'data_preprocessing.MeanImputation.DSBOX = dsbox.datapreprocessing.cleaner:MeanImputation',
              'normalization.IQRScaler.DSBOX = dsbox.datapreprocessing.cleaner:IQRScaler',
              'data_cleaning.Labeler.DSBOX = dsbox.datapreprocessing.cleaner:Labler',
              'normalization.Denormalize.DSBOX = dsbox.datapreprocessing.cleaner:Denormalize',
              'schema_discovery.Profiler.DSBOX = dsbox.datapreprocessing.cleaner:Profiler',
              'data_cleaning.FoldColumns.DSBOX = dsbox.datapreprocessing.cleaner:FoldColumns',
              # 'dsbox.Voter = dsbox.datapreprocessing.cleaner:Voter',
              'data_preprocessing.VerticalConcat.DSBOX = dsbox.datapostprocessing:VerticalConcat',
              'data_preprocessing.EnsembleVoting.DSBOX = dsbox.datapostprocessing:EnsembleVoting',
              'data_preprocessing.Unfold.DSBOX = dsbox.datapostprocessing:Unfold',
              'data_preprocessing.Splitter.DSBOX = dsbox.datapreprocessing.cleaner:Splitter',
              'data_preprocessing.HorizontalConcat.DSBOX = dsbox.datapostprocessing:HorizontalConcat',
            #   'data_augmentation.Augmentation.DSBOX = dsbox.datapreprocessing.cleaner:DatamartAugmentation',
            #   'data_augmentation.QueryDataframe.DSBOX = dsbox.datapreprocessing.cleaner:QueryFromDataframe',
            #   'data_augmentation.Join.DSBOX = dsbox.datapreprocessing.cleaner:DatamartJoin',
              'data_transformation.ToNumeric.DSBOX = dsbox.datapreprocessing.cleaner:ToNumeric'
          ],
      },
      cmdclass={
          'install': PostInstallCommand
      })
