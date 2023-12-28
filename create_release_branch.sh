#!/usr/bin/env bash

set -e

if [ $# -eq 0 ]; then
    echo "please write the type of release you would like to perform (major, minor, patch, or a specific version)"
    exit 1
fi

git fetch origin
git checkout develop
git pull origin develop

poetry version $1
version=$(poetry version --short)
message="Portfolyo Release $version"

git checkout -b $version

git add pyproject.toml
git commit -m "$message"
git push origin $version