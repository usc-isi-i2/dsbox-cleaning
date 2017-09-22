from setuptools import setup

setup(name='dsbox-datacleaning',
      version='0.2.0',
      url='https://github.com/usc-isi-i2/dsbox-cleaning.git',
      maintainer_email='kyao@isi.edu',
      maintainer='Ke-Thia Yao',
      description='DSBox data preprocessing tools for cleaning data',
      license='MIT',
      packages=['dsbox', 'dsbox.datapreprocessing', 'dsbox.datapreprocessing.cleaner'],
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=[
          'scipy>=0.19.0', 'numpy>=1.11.1', 'pandas>=0.20.1', 'langdetect>=1.0.7',
          'scikit-learn>=0.18.0', 'python-dateutil==2.5.2', 'six>=1.10.0', 'fancyimpute',
          'stopit'
      ]
)
