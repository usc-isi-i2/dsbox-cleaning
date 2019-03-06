#!/bin/sh

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_json_files() {
  git checkout -b primitive_repo
  git add .
  git commit -a --message "Travis build: $TRAVIS_BUILD_NUMBER"
}

upload_files() {
  git remote add upstream https://${GH_TOKEN}@github.com:usc-isi-i2/dsbox-cleaning.git
  git push --quiet --set-upstream upstream primitive_repo
}

setup_git
commit_json_files
upload_files