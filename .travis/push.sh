#!/bin/bash

if [[ $TRAVIS_BRANCH != "master" && $TRAVIS_BRANCH != "devel" ]];then
  echo "We're not in either master or devel branch."
  echo "Will not push generate pipelines json files or primitive json files"
  # analyze current branch and react accordingly
  exit 0
fi

python ../dsbox/generate-primitive-json.py
cd dsbox-unit-test-datasets
git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"

if [[ $TRAVIS_BRANCH == "master" ]];then
  echo "We're in master branch, will push generate json files to."
  echo "https://github.com/usc-isi-i2/dsbox-unit-test-datasets/tree/primitive_repo_cleaner_master"
  git checkout -b primitive_repo_cleaner_master
  rm -rf *
  mv ../output .
  git add .
  git commit -a --message "auto_generated_files"
  git remote add upstream https://${GH_TOKEN}@github.com/usc-isi-i2/dsbox-unit-test-datasets.git
  git push -f --quiet --set-upstream origin primitive_repo_cleaner_master

elif [[ $TRAVIS_BRANCH == "devel" ]];then
  echo "We're in devel branch, will push generate json files to."
  echo "https://github.com/usc-isi-i2/dsbox-unit-test-datasets/tree/primitive_repo_cleaner_devel"
  git checkout -b primitive_repo_cleaner_devel
  rm -rf *
  mv ../output .
  git add .
  git commit -a --message "auto_generated_files"
  git remote add upstream https://${GH_TOKEN}@github.com/usc-isi-i2/dsbox-unit-test-datasets.git
  git push -f --quiet --set-upstream origin primitive_repo_cleaner_devel
else
  echo "We're in ${TRAVIS_BRANCH} branch, will push generate json files to."
  echo "https://github.com/usc-isi-i2/dsbox-unit-test-datasets/tree/primitive_repo_cleaner_${TRAVIS_BRANCH}"
  git checkout -b primitive_repo_cleaner_${TRAVIS_BRANCH}
  rm -rf *
  mv ../output .
  git add .
  git commit -a --message "auto_generated_files"
  git remote add upstream https://${GH_TOKEN}@github.com/usc-isi-i2/dsbox-unit-test-datasets.git
  git push -f --quiet --set-upstream origin primitive_repo_cleaner_devel
fi
