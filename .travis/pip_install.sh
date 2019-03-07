#!/bin/bash
pip install -e git+https://gitlab.com/datadrivendiscovery/d3m@master#egg=d3m --progress-bar off
pip install -e git+https://gitlab.com/datadrivendiscovery/common-primitives@master#egg=common-primitives --progress-bar off
pip install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap@dist#egg=sklearn-wrap --progress-bar off
pip install -e . --progress-bar off
pip install -e git+https://github.com/brekelma/dsbox_corex@master#egg=dsbox_corex --progress-bar off
if [[ $TRAVIS_BRANCH == "master" ]];then
  echo "We're in master branch, will install dsbox-featurizer on master branch, too."
  pip install -e git+https://github.com/usc-isi-i2/dsbox-featurizer@master#egg=dsbox-featurizer --progress-bar off
else
  echo "We're not in master branch, will install dsbox-featurizer on devel branch."
  pip install -e git+https://github.com/usc-isi-i2/dsbox-featurizer@devel#egg=dsbox-featurizer --progress-bar off
fi
